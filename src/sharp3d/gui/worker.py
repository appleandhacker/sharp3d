"""GPU pipeline worker for the sharp3d GUI — multiprocess architecture.

Why a separate process instead of a QThread:
  Running the torch.compile + triton + gsplat CUDA stack inside a QThread on
  Windows corrupts the heap (0xC0000374) and crashes the whole app. The exact
  same pipeline runs stably in a plain process (proven by the CLI/scripts).
  So the heavy work lives in a Qt-free child process; the GUI talks to it via
  multiprocessing queues. This keeps the GUI responsive AND crash-isolated.

Layout:
  - _PipelineWorker : the real pipeline (model, predict, render, encode).
                      Runs in the child process. No Qt. Emits results by
                      calling a respond callback that feeds the response queue.
  - _child_main     : child-process entry loop (dispatch requests).
  - EngineProcess   : a QObject living on the GUI thread. Sends requests to
                      the child and polls the response queue, re-emitting
                      results as Qt signals. Exposes the same signal/method
                      surface the tabs already use.
"""

import gc
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal


def _quat_multiply(q1, q2):
    """Quaternion multiplication (wxyz convention), batched.

    Args:
        q1: [..., 4] (w, x, y, z)
        q2: [..., 4] (w, x, y, z)
    Returns:
        [..., 4] product q1 * q2
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    import torch
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _compute_render_face_size(eye_w: int, output_projection: str) -> int:
    """Compute adaptive cubemap render face size from output resolution.

    Each cubemap face covers 90° FOV:
      - 180° output: face covers 90/180 = 1/2 of width → eye_w
      - 360° output: face covers 90/360 = 1/4 of width → eye_w / 2

    Minimum 2048, rounded up to multiple of 256 for GPU efficiency.
    """
    if output_projection == "equirect180":
        ideal = eye_w
    else:  # equirect360
        ideal = eye_w // 2
    # Floor at 2048, round UP to multiple of 256
    clamped = max(2048, ideal)
    return (clamped + 255) // 256 * 256


# ===========================================================================
# Child-process side (no Qt)
# ===========================================================================
class _PipelineWorker:
    """The heavy pipeline. Lives in the child process."""

    def __init__(self, respond, cancel_event):
        self._respond = respond          # callable(name, args_tuple)
        self._cancel_event = cancel_event
        self._pipeline = None
        self._compiled = None
        self._device = None
        self._torch = None
        self._gaussians = None
        self._f_px = 1.0
        self._orig_w = 0
        self._orig_h = 0

    # ---- lazy model loading --------------------------------------------
    def preload(self):
        """Load + compile everything ahead of time (called at app startup).

        Same as the lazy path, but kicked off before the user picks a file so
        the one-time model/compile cost overlaps with natural UI idle time.
        """
        try:
            self._ensure_pipeline()
        except Exception as exc:  # noqa: BLE001
            self._respond("error", (f"预加载失败: {exc}",))

    def _ensure_pipeline(self, perf_mode="quality"):
        # Rebuild if mode changed
        if self._pipeline is not None and getattr(self, '_perf_mode_active', None) != perf_mode:
            self._respond("status", ("性能模式已切换，正在重建管线…",))
            self._pipeline = None
            self._compiled = None
            import torch
            torch._dynamo.reset()
            torch.cuda.empty_cache()

        if self._pipeline is not None:
            return
        self._perf_mode_active = perf_mode
        self._respond("model_loading", ())

        import torch
        from sharp3d.predict import SharpPredictor
        from pathlib import Path as _P

        self._device = torch.device("cuda")
        self._torch = torch
        cache_dir = _P(__file__).resolve().parents[3] / ".cache"

        def _progress(stage, pct):
            self._respond("model_load_progress", (stage, pct))
            self._respond("status", (stage + "…",))

        sp = SharpPredictor(
            device=self._device,
            perf_mode=perf_mode,
            cache_dir=cache_dir,
            progress_cb=_progress,
        )
        self._pipeline = sp.predictor
        self._compiled = sp  # SharpPredictor is callable (predict_fn)
        self._respond("model_ready", ())
        self._respond("status", ("模型就绪",))

    # ---- prepare: predict + unproject -> cache gaussians ----------------
    def prepare(self, path, frame_idx):
        try:
            self._ensure_pipeline()
            torch = self._torch
            from sharp.utils import io as sharp_io
            from sharp3d.unproject import prepare_input, fast_unproject, INTERNAL_SHAPE

            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
            p = Path(path)
            if p.suffix.lower() in video_exts:
                from sharp3d.hdr import FrameReader
                reader = FrameReader(p)
                frame = reader.read_frame(frame_idx)
                f_px = frame.shape[1] * 1.2
                image_np = frame
            else:
                image_np, _, f_px = sharp_io.load_rgb(p)

            h, w = image_np.shape[:2]
            self._f_px = f_px
            self._orig_w, self._orig_h = w, h

            img_r, df, ir, _ = prepare_input(image_np, f_px, self._device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                g_ndc = self._compiled(img_r, df)
            self._gaussians = fast_unproject(
                g_ndc, torch.eye(4, device=self._device), ir, INTERNAL_SHAPE,
                decompose_method="analytical",
            )
            torch.cuda.synchronize()

            n_g = self._gaussians.mean_vectors.numel() // 3
            self._respond("prepared", ({"width": w, "height": h,
                                        "n_gaussians": n_g, "f_px": f_px},))
            self._respond("status", (f"已重建 3D 场景 · {w}×{h}",))
            del g_ndc, img_r
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            self._respond("error", (f"准备失败: {exc}",))

    # ---- preview render from cached gaussians ---------------------------
    def render_preview(self, ipd_mm, convergence, strength, preview_width):
        if self._gaussians is None:
            return
        try:
            torch = self._torch
            from sharp3d.render import render_sbs, _compute_focus_depth_gpu
            ipd_scene = (ipd_mm / 1000.0) * strength
            q = convergence if convergence > 0 else 0.50
            conv = _compute_focus_depth_gpu(self._gaussians.mean_vectors, q_focus=q)
            with torch.no_grad():
                sbs, _ = render_sbs(
                    self._gaussians, self._f_px, self._orig_w, self._orig_h,
                    ipd=ipd_scene, convergence=conv, render_width=preview_width,
                )
            torch.cuda.synchronize()
            self._respond("preview_ready", (sbs.cpu().numpy(),))
        except Exception as exc:  # noqa: BLE001
            self._respond("error", (f"预览渲染失败: {exc}",))

    # ---- full conversion ------------------------------------------------
    def convert(self, opts):
        try:
            # VR panoramic mode
            if opts.get("mode") == "vr":
                self._convert_vr(opts)
                return

            self._ensure_pipeline(opts.get("perf_mode", "quality"))
            torch = self._torch
            from sharp.utils import io as sharp_io
            from sharp3d.unproject import prepare_input, fast_unproject, INTERNAL_SHAPE
            from sharp3d.render import render_sbs

            self._cancel_event.clear()
            path = Path(opts["input"])
            out = Path(opts["output"])
            ipd_scene = (opts["ipd_mm"] / 1000.0) * opts["strength"]
            conv_q = opts["convergence"] if opts["convergence"] > 0 else None
            method = opts.get("decompose", "analytical")

            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
            is_video = path.suffix.lower() in video_exts
            if is_video:
                self._convert_video(path, out, opts, ipd_scene, conv_q, method,
                                    prepare_input, fast_unproject, render_sbs,
                                    INTERNAL_SHAPE, torch)
            else:
                self._convert_image(path, out, opts, ipd_scene, conv_q, method,
                                    prepare_input, fast_unproject, render_sbs,
                                    INTERNAL_SHAPE, torch, sharp_io)
        except Exception as exc:  # noqa: BLE001
            self._respond("error", (f"转换失败: {exc}",))

    def _convert_vr(self, opts):
        """VR panoramic conversion: equirect/fisheye → stereo 3D VR."""
        self._ensure_pipeline(opts.get("perf_mode", "quality"))
        torch = self._torch
        from PIL import Image
        from sharp3d.unproject import prepare_input, fast_unproject, INTERNAL_SHAPE
        from sharp3d.projection import equirect_to_cubemap, fisheye_to_cubemap
        from sharp3d.render_vr import render_vr_stereo

        self._cancel_event.clear()
        path = Path(opts["input"])
        out = Path(opts["output"])
        ipd_scene = (opts["ipd_mm"] / 1000.0) * opts["strength"]
        renderer = opts.get("renderer", "higs")
        output_projection = opts.get("output_projection", "equirect180")
        stereo_layout = opts.get("stereo_layout", "sbs")
        eye_w = opts.get("eye_width", 4096)
        eye_h = opts.get("eye_height", 4096)
        input_projection = opts.get("input_projection", "auto")
        ftheta_coeffs = opts.get("ftheta_coeffs")

        # Determine input type
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        is_video = path.suffix.lower() in video_exts

        if is_video:
            self._convert_vr_video(path, out, opts, ipd_scene, renderer,
                                   output_projection, stereo_layout,
                                   eye_w, eye_h, input_projection,
                                   ftheta_coeffs, torch,
                                   prepare_input, fast_unproject,
                                   equirect_to_cubemap, fisheye_to_cubemap,
                                   render_vr_stereo, INTERNAL_SHAPE)
        else:
            self._convert_vr_image(path, out, opts, ipd_scene, renderer,
                                   output_projection, stereo_layout,
                                   eye_w, eye_h, input_projection,
                                   ftheta_coeffs, torch,
                                   prepare_input, fast_unproject,
                                   equirect_to_cubemap, fisheye_to_cubemap,
                                   render_vr_stereo, INTERNAL_SHAPE)

    def _convert_vr_image(self, path, out, opts, ipd_scene, renderer,
                          output_projection, stereo_layout,
                          eye_w, eye_h, input_projection,
                          ftheta_coeffs, torch,
                          prepare_input, fast_unproject,
                          equirect_to_cubemap, fisheye_to_cubemap,
                          render_vr_stereo, INTERNAL_SHAPE):
        """Single-image VR conversion."""
        from PIL import Image
        import numpy as np

        t_start = time.time()

        # Load input image
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        device = self._device

        # Convert input to tensor [H, W, 3] float [0, 1]
        img_t = torch.from_numpy(img_np).float().to(device) / 255.0

        # Auto-detect projection if needed
        proj = input_projection
        if proj == "auto":
            aspect = w / h
            if 1.9 < aspect < 2.1:
                proj = "equirect360"
            elif 0.9 < aspect < 1.1:
                proj = "equirect180"
            else:
                proj = "equirect360"  # default fallback

        # Extract faces from input (overlapping FOV for seam reduction)
        from sharp3d.projection import (OVERLAP_FOV_SCALE, OVERLAP_KEEP_ANGLE_DEG,
                                        filter_gaussians_by_angle,
                                        angular_opacity_weight)
        face_size = 1536  # match SHARP internal resolution

        if proj == "equirect360":
            # Full 6-face cubemap for 360° equirectangular
            from sharp3d.projection import (get_cubemap_cameras,
                                            _look_at_rotation, _FACE_DEFS)
            faces = equirect_to_cubemap(img_t, face_size,
                                        fov_scale=OVERLAP_FOV_SCALE)
            viewmats, _ = get_cubemap_cameras(face_size, device)
            face_forwards = [fd[0] for fd in _FACE_DEFS]
            n_faces = 6
        else:
            # Optimal 4-axis hemisphere coverage for fisheye / equirect180
            from sharp3d.projection import (get_hemisphere_cameras,
                                            _HEMISPHERE_AXES)
            if proj.startswith("fisheye"):
                from sharp3d.projection import fisheye_to_hemisphere
                model_map = {
                    "fisheye_equidistant": "equidistant",
                    "fisheye_equisolid": "equisolid",
                    "fisheye_orthographic": "orthographic",
                    "fisheye_stereographic": "stereographic",
                    "fisheye_ftheta": "ftheta",
                }
                model = model_map.get(proj, "equidistant")
                faces = fisheye_to_hemisphere(img_t, face_size, model=model,
                                              coeffs=ftheta_coeffs,
                                              fov_scale=OVERLAP_FOV_SCALE)
            else:
                # equirect180
                from sharp3d.projection import equirect_to_hemisphere
                faces = equirect_to_hemisphere(img_t, face_size,
                                               fov_scale=OVERLAP_FOV_SCALE)
            viewmats, _ = get_hemisphere_cameras(face_size, device)
            face_forwards = [ax[0] for ax in _HEMISPHERE_AXES]
            n_faces = 4

        total_steps = n_faces + 6  # prediction faces + 6 render milestones

        # Predict depth + unproject for each face → merge Gaussians
        from sharp3d.quaternion import quat_from_rotmat_gpu

        all_means = []
        all_quats = []
        all_scales = []
        all_opacities = []
        all_colors = []

        eye4 = torch.eye(4, device=device)
        f_px = face_size / (2.0 * OVERLAP_FOV_SCALE)
        # Seam angle: boundary between adjacent faces
        seam_deg = 45.0 if n_faces == 6 else 35.3  # cubemap vs hemisphere

        for i in range(n_faces):
            if self._cancel_event.is_set():
                self._respond("convert_done", ({"cancelled": True},))
                return

            # Face image — GPU-direct (prepare_input accepts GPU tensor)
            img_r, df, ir, _ = prepare_input(faces[i], f_px, device)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                g_ndc = self._compiled(img_r, df)

            g = fast_unproject(g_ndc, eye4, ir,
                               INTERNAL_SHAPE, decompose_method="analytical")

            # Squeeze batch dim: [1, N, ...] → [N, ...]
            # Clone to detach from CUDA Graphs output buffers (reused across runs)
            means = g.mean_vectors.squeeze(0).clone()
            quats_local = g.quaternions.squeeze(0).clone()
            scales = g.singular_values.squeeze(0).clone()
            opacities = (g.opacities.squeeze(0) if g.opacities.dim() == 2 else g.opacities).clone()
            colors = g.colors.squeeze(0).clone()

            # Transform Gaussians from face-local to world space
            R = viewmats[i, :3, :3]  # world-to-camera rotation
            R_inv = R.T  # camera-to-world

            means_world = means @ R_inv.T
            # Rotate quaternions: q_world = q_rot * q_local
            q_rot = quat_from_rotmat_gpu(R_inv.unsqueeze(0))[0]  # [4]

            # Quaternion multiplication: q_world = q_rot * q_local
            quats_world = _quat_multiply(q_rot.unsqueeze(0), quats_local)

            # Center-weighted opacity falloff (smooth seam blending)
            face_fwd = face_forwards[i].to(device)
            weight = angular_opacity_weight(means_world, face_fwd,
                                            inner_deg=seam_deg)
            keep = weight > 0.01
            w_keep = weight[keep]
            opac_keep = opacities[keep]
            if opac_keep.dim() > w_keep.dim():
                w_keep = w_keep.unsqueeze(-1)
            all_means.append(means_world[keep])
            all_quats.append(quats_world[keep])
            all_scales.append(scales[keep])
            all_opacities.append(opac_keep * w_keep)
            all_colors.append(colors[keep])

            self._respond("convert_progress",
                          (i + 1, total_steps, 0.0, time.time() - t_start))

        # Merge all Gaussians
        from sharp.utils.gaussians import Gaussians3D
        merged = Gaussians3D(
            mean_vectors=torch.cat(all_means, dim=0),
            singular_values=torch.cat(all_scales, dim=0),
            quaternions=torch.cat(all_quats, dim=0),
            colors=torch.cat(all_colors, dim=0),
            opacities=torch.cat(all_opacities, dim=0),
        )

        # PLY export (world-space Gaussians) — non-fatal if it fails
        if opts.get("ply"):
            try:
                from sharp.utils.gaussians import save_ply
                # save_ply expects [1, N, C] (batch dim) — it calls .flatten(0,1)
                opac = merged.opacities
                if opac.dim() == 1:
                    opac = opac.unsqueeze(-1)  # [N] → [N,1]
                merged_ply = Gaussians3D(
                    mean_vectors=merged.mean_vectors.unsqueeze(0),
                    singular_values=merged.singular_values.unsqueeze(0),
                    quaternions=merged.quaternions.unsqueeze(0),
                    colors=merged.colors.unsqueeze(0),
                    opacities=opac.unsqueeze(0),
                )
                save_ply(merged_ply, 1.0, (1, 1), out.with_suffix(".ply"))
            except Exception as e:
                self._respond("status", (f"PLY导出失败(不影响转换): {e}",))

        # Render VR stereo (adaptive face size based on output resolution)
        render_face = _compute_render_face_size(eye_w, output_projection)

        def _render_progress(step, total):
            # Render reports 1-6, map to consecutive steps after prediction
            self._respond("convert_progress",
                          (n_faces + step, total_steps, 0.0, time.time() - t_start))

        result = render_vr_stereo(
            merged,
            ipd=ipd_scene,
            face_size=render_face,
            out_w=eye_w,
            out_h=eye_h,
            output_projection=output_projection,
            stereo_layout=stereo_layout,
            renderer=renderer,
            device=device,
            progress_cb=_render_progress,
        )

        # Save
        Image.fromarray(result.cpu().numpy()).save(out)

        # Depth panorama output
        if opts.get("depth"):
            from sharp3d.projection import cubemap_to_equirect, cubemap_to_equirect180
            from gsplat.rendering import rasterization as _rast
            from sharp3d.projection import get_cubemap_cameras as _gcc

            depth_map_fn = cubemap_to_equirect180 if output_projection == "equirect180" else cubemap_to_equirect
            viewmats_d, Ks_d = _gcc(render_face, device)
            # Render depth from left eye (no stereo offset for depth)
            with torch.no_grad():
                rendered_d, _, meta_d = _rast(
                    means=merged.mean_vectors,
                    quats=merged.quaternions,
                    scales=merged.singular_values,
                    opacities=merged.opacities,
                    colors=merged.colors,
                    viewmats=viewmats_d,
                    Ks=Ks_d,
                    width=render_face, height=render_face,
                    render_mode="RGB+D",
                )
            # RGB+D mode: rendered_d shape [6, H, W, 4] (RGB + depth)
            depths = rendered_d[:, :, :, 3:4].permute(0, 3, 1, 2)  # [6, 1, H, W]
            depths_3ch = depths.expand(-1, 3, -1, -1)  # [6, 3, H, W] for assembly
            depth_equirect = depth_map_fn(depths_3ch, eye_w, eye_h)  # [H, W, 3]
            # Normalize depth: near=bright, far=dark, log scale for contrast
            d_valid = depth_equirect[depth_equirect > 0]
            if d_valid.numel() > 0:
                d_min = d_valid.min()
                d_max = d_valid.max()
                if d_max > d_min:
                    # Log-scale mapping for better near-range contrast
                    depth_log = torch.log(depth_equirect.clamp(min=d_min) / d_min + 1e-6)
                    log_max = torch.log(d_max / d_min + 1e-6)
                    # Invert: near (small depth) → bright (255), far → dark (0)
                    depth_vis = ((1.0 - depth_log / log_max) * 255).clamp(0, 255).to(torch.uint8)
                else:
                    depth_vis = torch.full_like(depth_equirect, 128, dtype=torch.uint8)
            else:
                depth_vis = torch.zeros_like(depth_equirect, dtype=torch.uint8)
            # Zero-depth pixels (no coverage) → black
            depth_vis[depth_equirect <= 0] = 0
            depth_out = out.with_stem(out.stem + "_depth")
            Image.fromarray(depth_vis.cpu().numpy()).save(depth_out)

        del merged, faces, result
        torch.cuda.empty_cache()
        gc.collect()

        self._respond("convert_progress", (total_steps, total_steps, 1.0, time.time() - t_start))
        elapsed = time.time() - t_start
        self._respond("convert_done", ({
            "output": str(out), "elapsed": elapsed,
            "fps": 1.0 / max(elapsed, 1e-6),
            "n_frames": 1,
        },))

    def _convert_vr_video(self, path, out, opts, ipd_scene, renderer,
                          output_projection, stereo_layout,
                          eye_w, eye_h, input_projection,
                          ftheta_coeffs, torch,
                          prepare_input, fast_unproject,
                          equirect_to_cubemap, fisheye_to_cubemap,
                          render_vr_stereo, INTERNAL_SHAPE):
        """Video VR conversion — frame-by-frame cubemap predict + render."""
        from sharp3d.hdr import FrameReader, FFMPEG, hdr_to_sdr_filter
        from sharp3d.video import VideoWriter, resolve_encoder
        from sharp3d.projection import (OVERLAP_FOV_SCALE,
                                        OVERLAP_KEEP_ANGLE_DEG,
                                        filter_gaussians_by_angle,
                                        angular_opacity_weight)
        from sharp3d.quaternion import quat_from_rotmat_gpu
        from sharp.utils.gaussians import Gaussians3D
        import queue as _queue
        import threading
        import subprocess as _sp

        reader = FrameReader(path)
        n = reader.n_frames
        device = self._device

        # Auto-detect input projection
        proj = input_projection
        if proj == "auto":
            aspect = reader.width / reader.height
            if 1.9 < aspect < 2.1:
                proj = "equirect360"
            elif 0.9 < aspect < 1.1:
                proj = "equirect180"
            else:
                proj = "equirect360"

        # Output video dimensions (packed stereo)
        if stereo_layout == "sbs":
            vid_w, vid_h = eye_w * 2, eye_h
        else:  # tb
            vid_w, vid_h = eye_w, eye_h * 2

        # Encoder setup
        codec = opts.get("codec", "av1")
        crf = opts.get("crf", 20)
        out_fps = opts.get("out_fps") or reader.fps
        enc = resolve_encoder(codec, vid_w, vid_h)
        is_gpu = "nvenc" in enc
        label = "GPU" if is_gpu else "CPU"
        self._respond("status", (
            f"编码器: {enc} ({label}) · 输出 {vid_w}×{vid_h}",))

        writer = VideoWriter(out, fps=out_fps, width=vid_w, height=vid_h,
                             codec=codec, crf=crf)

        # Cubemap prediction setup (fixed across frames)
        pred_face_size = 1536  # SHARP internal resolution
        render_face = _compute_render_face_size(eye_w, output_projection)
        f_px = pred_face_size / (2.0 * OVERLAP_FOV_SCALE)
        eye4 = torch.eye(4, device=device)
        seam_deg = 45.0  # updated below based on face layout

        # Determine face layout based on input projection
        if proj == "equirect360":
            from sharp3d.projection import get_cubemap_cameras, _FACE_DEFS
            viewmats, _ = get_cubemap_cameras(pred_face_size, device)
            face_forwards = [fd[0] for fd in _FACE_DEFS]
            n_faces = 6
            use_hemisphere = False
        else:
            from sharp3d.projection import (get_hemisphere_cameras,
                                            _HEMISPHERE_AXES)
            viewmats, _ = get_hemisphere_cameras(pred_face_size, device)
            face_forwards = [ax[0] for ax in _HEMISPHERE_AXES]
            n_faces = 4
            use_hemisphere = True
            seam_deg = 35.3

        # Fisheye model map (invariant)
        model_map = {
            "fisheye_equidistant": "equidistant",
            "fisheye_equisolid": "equisolid",
            "fisheye_orthographic": "orthographic",
            "fisheye_stereographic": "stereographic",
            "fisheye_ftheta": "ftheta",
        }

        # Decode prefetch thread
        frame_q: _queue.Queue = _queue.Queue(maxsize=4)
        frame_size = reader.width * reader.height * 3

        def _decode():
            vf_parts = []
            if reader.is_hdr:
                vf_parts.append(hdr_to_sdr_filter())
            vf = ["-vf", ",".join(vf_parts)] if vf_parts else []
            cmd = [FFMPEG, *reader._hwaccel(),
                   "-i", reader.path, *vf,
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL,
                             creationflags=0x08000000)
            try:
                while True:
                    if self._cancel_event.is_set():
                        break
                    raw = proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break
                    frame_q.put(
                        np.frombuffer(raw, dtype=np.uint8).reshape(
                            reader.height, reader.width, 3).copy())
            finally:
                proc.stdout.close()
                proc.wait()
                frame_q.put(None)

        decoder = threading.Thread(target=_decode, daemon=True)
        decoder.start()

        # Encode thread (overlaps encoding with GPU compute)
        encode_q: _queue.Queue = _queue.Queue(maxsize=3)

        def _encode_loop():
            while True:
                item = encode_q.get()
                if item is None:
                    break
                writer.append_frame(item)

        encode_thread = threading.Thread(target=_encode_loop, daemon=True)
        encode_thread.start()

        # Depth video (optional)
        want_depth = opts.get("depth", False)
        depth_writer = None
        if want_depth:
            from sharp3d.projection import cubemap_to_equirect, cubemap_to_equirect180
            from gsplat.rendering import rasterization as _rast
            depth_path = out.with_stem(out.stem + "_depth")
            depth_writer = VideoWriter(str(depth_path), fps=out_fps,
                                       width=eye_w, height=eye_h,
                                       codec="h264", crf=18)

        # Main conversion loop
        n_done = 0
        t_start = time.time()

        while True:
            frm = frame_q.get()
            if frm is None or self._cancel_event.is_set():
                break

            # Convert frame to tensor [H, W, 3] float [0, 1]
            img_t = torch.from_numpy(frm).float().to(device) / 255.0
            del frm

            # Extract faces (overlapping FOV)
            if use_hemisphere:
                from sharp3d.projection import (equirect_to_hemisphere,
                                                fisheye_to_hemisphere)
                if proj.startswith("fisheye"):
                    model = model_map.get(proj, "equidistant")
                    faces = fisheye_to_hemisphere(img_t, pred_face_size,
                                                  model=model,
                                                  coeffs=ftheta_coeffs,
                                                  fov_scale=OVERLAP_FOV_SCALE)
                else:
                    faces = equirect_to_hemisphere(img_t, pred_face_size,
                                                   fov_scale=OVERLAP_FOV_SCALE)
            else:
                if proj.startswith("equirect"):
                    faces = equirect_to_cubemap(img_t, pred_face_size,
                                                fov_scale=OVERLAP_FOV_SCALE)
                else:
                    model = model_map.get(proj, "equidistant")
                    faces = fisheye_to_cubemap(img_t, pred_face_size,
                                               model=model,
                                               coeffs=ftheta_coeffs,
                                               fov_scale=OVERLAP_FOV_SCALE)
            del img_t

            # Predict + unproject each face → merge Gaussians
            all_means = []
            all_quats = []
            all_scales = []
            all_opacities = []
            all_colors = []

            for i in range(n_faces):
                if self._cancel_event.is_set():
                    break
                # GPU-direct (prepare_input accepts GPU tensor)
                img_r, df, ir, _ = prepare_input(faces[i], f_px, device)

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    g_ndc = self._compiled(img_r, df)

                g = fast_unproject(g_ndc, eye4, ir,
                                   INTERNAL_SHAPE, decompose_method="analytical")

                means = g.mean_vectors.squeeze(0).clone()
                quats_local = g.quaternions.squeeze(0).clone()
                scales = g.singular_values.squeeze(0).clone()
                opacities = (g.opacities.squeeze(0) if g.opacities.dim() == 2
                             else g.opacities).clone()
                colors = g.colors.squeeze(0).clone()

                R = viewmats[i, :3, :3]
                R_inv = R.T
                means_world = means @ R_inv.T
                q_rot = quat_from_rotmat_gpu(R_inv.unsqueeze(0))[0]
                quats_world = _quat_multiply(q_rot.unsqueeze(0), quats_local)

                # Center-weighted opacity falloff (smooth seam blending)
                face_fwd = face_forwards[i].to(device)
                weight = angular_opacity_weight(means_world, face_fwd,
                                                inner_deg=seam_deg)
                keep = weight > 0.01
                w_keep = weight[keep]
                opac_keep = opacities[keep]
                if opac_keep.dim() > w_keep.dim():
                    w_keep = w_keep.unsqueeze(-1)
                all_means.append(means_world[keep])
                all_quats.append(quats_world[keep])
                all_scales.append(scales[keep])
                all_opacities.append(opac_keep * w_keep)
                all_colors.append(colors[keep])

            if self._cancel_event.is_set():
                del faces
                break

            merged = Gaussians3D(
                mean_vectors=torch.cat(all_means, dim=0),
                singular_values=torch.cat(all_scales, dim=0),
                quaternions=torch.cat(all_quats, dim=0),
                colors=torch.cat(all_colors, dim=0),
                opacities=torch.cat(all_opacities, dim=0),
            )
            del faces, all_means, all_quats, all_scales, all_opacities, all_colors

            # Render stereo VR frame
            result = render_vr_stereo(
                merged,
                ipd=ipd_scene,
                face_size=render_face,
                out_w=eye_w,
                out_h=eye_h,
                output_projection=output_projection,
                stereo_layout=stereo_layout,
                renderer=renderer,
                device=device,
            )

            # Encode (async)
            encode_q.put(result.cpu().numpy())
            del result

            # Depth output (optional)
            if depth_writer is not None:
                from sharp3d.projection import (cubemap_to_equirect as _c2e,
                                               cubemap_to_equirect180 as _c2e180)
                from gsplat.rendering import rasterization as _rast
                depth_map_fn = _c2e180 if output_projection == "equirect180" else _c2e
                viewmats_d, Ks_d = get_cubemap_cameras(render_face, device)
                with torch.no_grad():
                    rendered_d, _, _ = _rast(
                        means=merged.mean_vectors,
                        quats=merged.quaternions,
                        scales=merged.singular_values,
                        opacities=merged.opacities,
                        colors=merged.colors,
                        viewmats=viewmats_d,
                        Ks=Ks_d,
                        width=render_face, height=render_face,
                        render_mode="RGB+D",
                    )
                depths = rendered_d[:, :, :, 3:4].permute(0, 3, 1, 2)
                depths_3ch = depths.expand(-1, 3, -1, -1)
                depth_equirect = depth_map_fn(depths_3ch, eye_w, eye_h)
                d_valid = depth_equirect[depth_equirect > 0]
                if d_valid.numel() > 0:
                    d_min = d_valid.min()
                    d_max = d_valid.max()
                    if d_max > d_min:
                        depth_log = torch.log(
                            depth_equirect.clamp(min=d_min) / d_min + 1e-6)
                        log_max = torch.log(d_max / d_min + 1e-6)
                        depth_vis = ((1.0 - depth_log / log_max) * 255
                                     ).clamp(0, 255).to(torch.uint8)
                    else:
                        depth_vis = torch.full_like(depth_equirect, 128,
                                                    dtype=torch.uint8)
                else:
                    depth_vis = torch.zeros_like(depth_equirect, dtype=torch.uint8)
                depth_vis[depth_equirect <= 0] = 0
                depth_writer.append_frame(depth_vis.cpu().numpy())
                del rendered_d, depths, depth_equirect, depth_vis

            del merged
            torch.cuda.empty_cache()

            n_done += 1
            elapsed = time.time() - t_start
            avg_fps = n_done / elapsed if elapsed > 0 else 0.0
            self._respond("convert_progress", (n_done, n, avg_fps, elapsed))

            if n_done % 30 == 0:
                gc.collect()

        # Cleanup
        encode_q.put(None)
        encode_thread.join(timeout=60)
        while True:
            try:
                frame_q.get_nowait()
            except _queue.Empty:
                break
        decoder.join(timeout=5)

        keep_audio = opts.get("audio", True) and reader.has_audio
        source = path if keep_audio else None
        writer.close(source_video=source)
        if depth_writer is not None:
            depth_writer.close()

        total_elapsed = time.time() - t_start
        avg = n_done / total_elapsed if total_elapsed > 0 else 0.0
        self._respond("convert_done", ({
            "output": str(out), "elapsed": total_elapsed,
            "fps": avg, "n_frames": n_done,
            "size": (vid_w, vid_h),
            "cancelled": self._cancel_event.is_set(),
        },))

    def _convert_image(self, path, out, opts, ipd_scene, conv_q, method,
                       prepare_input, fast_unproject, render_sbs,
                       INTERNAL_SHAPE, torch, sharp_io):
        from PIL import Image
        from sharp3d.formats import output_size, pack as pack_stereo

        fmt = opts.get("format", "full_sbs")
        image_np, _, f_px = sharp_io.load_rgb(path)
        h, w = image_np.shape[:2]

        # Output resolution: custom width or scale fraction of source width.
        custom_w = opts.get("out_width")
        out_scale = opts.get("out_scale", 1.0)
        render_w = int(custom_w) if custom_w else int(round(w * out_scale))
        render_w = max(2, render_w + render_w % 2)

        img_r, df, ir, _ = prepare_input(image_np, f_px, self._device)

        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g_ndc = self._compiled(img_r, df)
        g = fast_unproject(g_ndc, torch.eye(4, device=self._device), ir,
                           INTERNAL_SHAPE, decompose_method=method)
        from sharp3d.render import _compute_focus_depth_gpu
        q = conv_q if conv_q else 0.50
        conv_dist = _compute_focus_depth_gpu(g.mean_vectors, q_focus=q)
        sbs, (sw, sh) = render_sbs(g, f_px, w, h, ipd=ipd_scene,
                                   convergence=conv_dist, render_width=render_w)
        packed = pack_stereo(fmt, sbs)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        Image.fromarray(packed.cpu().numpy()).save(out)

        if opts.get("depth"):
            from sharp3d.render import render_depth_map
            depth = render_depth_map(g, f_px, w, h)
            Image.fromarray(depth.cpu().numpy()).save(
                out.with_stem(out.stem + "_depth"))
        if opts.get("ply"):
            from sharp.utils.gaussians import save_ply
            save_ply(g, f_px, (h, w), out.with_suffix(".ply"))

        del g, g_ndc, sbs, packed
        torch.cuda.empty_cache()
        gc.collect()

        self._respond("convert_progress", (1, 1, 1.0 / elapsed, elapsed))
        self._respond("convert_done", ({
            "output": str(out), "elapsed": elapsed, "fps": 1.0 / elapsed,
            "n_frames": 1, "size": output_size(fmt, sw, sh),
        },))

    def _convert_video(self, path, out, opts, ipd_scene, conv_q, method,
                       prepare_input, fast_unproject, render_sbs,
                       INTERNAL_SHAPE, torch):
        from sharp3d.hdr import FrameReader, Hdr10Writer
        from sharp3d.video import VideoWriter, resolve_encoder
        from sharp3d.formats import output_size, pack as pack_stereo

        reader = FrameReader(path)
        n = reader.n_frames
        f_px = reader.width * 1.2

        fmt = opts.get("format", "full_sbs")

        # ── Output resolution: custom width or scale fraction of source ──
        custom_w = opts.get("out_width")          # per-eye width (px) or None
        out_scale = opts.get("out_scale", 1.0)    # fraction of source width
        if custom_w:
            render_w = int(custom_w)
        else:
            render_w = int(round(reader.width * out_scale))
        render_w = max(2, render_w + render_w % 2)
        # Per-eye render height follows the source aspect ratio (even), matching
        # render_sbs/_get_screen_resolution so the writer size stays consistent.
        render_h = int(round(reader.height * (render_w / reader.width)))
        render_h = max(2, render_h + render_h % 2)
        sw, sh = render_w, render_h
        if sh > 3000:                      # mirrors _get_screen_resolution
            sw, sh = sw // 2, sh // 2
        sw += sw % 2
        sh += sh % 2
        out_w, out_h = output_size(fmt, sw, sh)

        # ── Output frame rate (None → keep source fps) ──
        out_fps = opts.get("out_fps") or reader.fps

        # ── Frame-rate conversion ────────────────────────────────────────
        # Use ffmpeg's fps filter to decimate/duplicate at decode time.
        # This is critical for VRAM stability: the old approach decoded ALL
        # source frames (NVDEC full speed) then discarded half in Python,
        # causing memory pressure from unused frames flowing through the pipe.
        # With the fps filter, ffmpeg only outputs the frames we actually need.
        src_fps = reader.fps or out_fps
        needs_fps_change = (out_fps and src_fps and abs(out_fps - src_fps) > 1e-6)
        if needs_fps_change:
            out_count = max(1, int(round(n * out_fps / src_fps)))
        else:
            out_count = n

        # Show which encoder was selected (GPU vs CPU) for all codecs.
        codec = opts.get("codec", "h264")
        if not opts.get("hdr_output", False):
            enc = resolve_encoder(codec, out_w, out_h)
            is_gpu = "nvenc" in enc
            label = "GPU" if is_gpu else "CPU"
            self._respond("status", (
                f"编码器: {enc} ({label}) · 输出 {out_w}×{out_h}",))
            if not is_gpu:
                self._respond("status", (
                    f"输出 {out_w}×{out_h} 超过NVENC分辨率上限，"
                    f"已回退CPU编码 ({enc})，CPU占用会较高",))

        hdr_out = opts.get("hdr_output", False)
        if hdr_out:
            writer = Hdr10Writer(out, width=out_w, height=out_h,
                                 fps=out_fps, codec=opts.get("codec", "h265"),
                                 crf=opts.get("crf", 18))
        else:
            writer = VideoWriter(out, fps=out_fps, width=out_w, height=out_h,
                                 codec=opts.get("codec", "h264"),
                                 crf=opts.get("crf", 18))

        # ── Decode prefetch thread ──────────────────────────────────────
        # ffmpeg handles fps conversion via its fps filter, so the decode
        # thread only produces frames that will actually be processed.
        # Queue of 4 frames balances prefetch against RAM pressure.
        import queue as _queue
        import threading
        import subprocess as _sp

        frame_q: _queue.Queue = _queue.Queue(maxsize=4)
        frame_size = reader.width * reader.height * 3

        def _decode():
            # Build ffmpeg command with optional fps filter for decimation.
            # This replaces the old Python-side skip logic that caused VRAM
            # growth (NVDEC decoded all frames, half were discarded unused).
            from sharp3d.hdr import FFMPEG, hdr_to_sdr_filter
            vf_parts = []
            if reader.is_hdr:
                vf_parts.append(hdr_to_sdr_filter())
            if needs_fps_change:
                vf_parts.append(f"fps={out_fps}")
            vf = ["-vf", ",".join(vf_parts)] if vf_parts else []
            cmd = [FFMPEG, *reader._hwaccel(),
                   "-i", reader.path, *vf,
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL,
                             creationflags=0x08000000)
            try:
                while True:
                    if self._cancel_event.is_set():
                        break
                    raw = proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break
                    frame_q.put(
                        np.frombuffer(raw, dtype=np.uint8).reshape(
                            reader.height, reader.width, 3).copy())
            finally:
                proc.stdout.close()
                proc.wait()
                frame_q.put(None)

        decoder = threading.Thread(target=_decode, daemon=True)
        decoder.start()

        # ── Async H2D upload on side stream (overlaps with GPU compute) ──
        _side_stream = torch.cuda.Stream()

        def _prepare(frm):
            with torch.cuda.stream(_side_stream):
                prepared = prepare_input(frm, f_px, self._device,
                                         async_upload=True)
                upload_event = _side_stream.record_event()
            return prepared, upload_event

        # ── Encode thread (overlaps CPU encoding with GPU compute) ──────
        import queue as _queue
        encode_q: _queue.Queue = _queue.Queue(maxsize=3)

        def _encode_loop():
            while True:
                item = encode_q.get()
                if item is None:
                    break
                frame_np, is_hdr = item
                if is_hdr:
                    writer.write_frame(frame_np)
                else:
                    writer.append_frame(frame_np)

        encode_thread = threading.Thread(target=_encode_loop, daemon=True)
        encode_thread.start()

        # ── Conversion engine (predict + stabilize + render) ────────────
        from sharp3d.conversion import VideoConversionEngine
        stab_mode = opts.get("temporal_stabilize", "off")
        engine = VideoConversionEngine(
            predict_fn=self._compiled,
            device=self._device,
            f_px=f_px,
            fmt=fmt,
            ipd=ipd_scene,
            convergence_q=conv_q,
            decompose_method=method,
            stabilize_mode=stab_mode,
            render_width=render_w,
            edge_soften=opts.get("edge_soften", False),
        )

        # ── Optional depth/PLY export ────────────────────────────────────
        want_depth = opts.get("depth", False)
        want_ply = opts.get("ply", False)
        depth_writer = None
        if want_depth:
            depth_path = out.with_stem(out.stem + "_depth")
            depth_writer = VideoWriter(str(depth_path), fps=out_fps,
                                       width=render_w, height=render_h,
                                       codec="h264", crf=18)

        # ── Main conversion loop ────────────────────────────────────────
        # Every frame from the queue is processed (ffmpeg already selected the
        # correct frames via its fps filter). No Python-side skip logic.
        out_written = 0
        t_start = time.time()
        n_done = 0

        first = frame_q.get()
        try:
            if first is not None and not self._cancel_event.is_set():
                prepared, upload_ev = _prepare(first)
                del first

                while True:
                    # ── 预取下一帧（与当前帧 GPU 计算重叠）──────────
                    nxt_frm = frame_q.get()
                    if nxt_frm is not None:
                        nxt_prepared, nxt_ev = _prepare(nxt_frm)
                        del nxt_frm
                    else:
                        nxt_prepared, nxt_ev = None, None

                    # ── 等待当前帧上传完成，然后 GPU 处理 ──────────
                    upload_ev.synchronize()
                    img_r, df, ir, (w, h) = prepared

                    result = engine.process_frame(
                        img_r, df, ir, (w, h),
                        return_depth=want_depth,
                        return_gaussians=want_ply,
                    )

                    # Unpack results
                    if want_ply:
                        sbs_np, depth_np, g_world = result
                        from sharp.utils.gaussians import save_ply
                        ply_path = out.with_suffix("") / f"{out.stem}_{n_done:05d}.ply"
                        ply_path.parent.mkdir(parents=True, exist_ok=True)
                        save_ply(g_world, f_px, (h, w), ply_path)
                        if n_done == 0:
                            self._respond("status", (
                                f"PLY 序列导出中: {ply_path.parent.name}/",))
                        del g_world
                    elif want_depth:
                        sbs_np, depth_np = result
                    else:
                        sbs_np = result
                        depth_np = None

                    # ── Encode (async via encode thread) ────────────
                    encode_q.put((sbs_np, hdr_out))
                    out_written += 1

                    # Depth video (synchronous, lightweight)
                    if depth_writer is not None and depth_np is not None:
                        depth_writer.append_frame(depth_np)

                    del img_r, prepared, sbs_np, depth_np

                    n_done += 1
                    elapsed = time.time() - t_start
                    avg_fps = n_done / elapsed if elapsed > 0 else 0.0
                    self._respond("convert_progress",
                                  (out_written, out_count, avg_fps, elapsed))

                    # Periodic CPU GC every 60 frames prevents RAM accumulation.
                    if n_done % 60 == 0:
                        gc.collect()

                    if self._cancel_event.is_set():
                        break
                    if nxt_prepared is None:
                        break

                    # ── 下一帧已预取，直接进入下一轮 ──────────────
                    prepared, upload_ev = nxt_prepared, nxt_ev
        finally:
            # Stop encode thread (flush remaining frames).
            encode_q.put(None)
            encode_thread.join(timeout=30)
            # Always clean up decoder thread (prevents leaked ffmpeg process).
            while True:
                try:
                    frame_q.get_nowait()
                except _queue.Empty:
                    break
            decoder.join(timeout=5)
            torch.cuda.empty_cache()

        keep_audio = opts.get("audio", True) and reader.has_audio
        source = path if keep_audio else None
        if hdr_out:
            writer.close(audio_source=source)
        else:
            writer.close(source_video=source)
        if depth_writer is not None:
            depth_writer.close()

        total_elapsed = time.time() - t_start
        avg = n_done / total_elapsed if total_elapsed > 0 else 0.0
        self._respond("convert_done", ({
            "output": str(out), "elapsed": total_elapsed,
            "fps": avg,
            "n_frames": n_done, "size": (out_w, out_h),
            "cancelled": self._cancel_event.is_set(), "hdr": hdr_out,
        },))

    # ---- 2.5D parallax animation ----------------------------------------
    def render_anim(self, opts):
        if self._gaussians is None:
            self._respond("error", ("请先加载一张图片",))
            return
        try:
            torch = self._torch
            from sharp.utils import camera as sharp_camera
            from sharp3d.render import render_single

            self._cancel_event.clear()
            params = sharp_camera.TrajectoryParams(
                type=opts["type"], max_disparity=opts["max_disparity"],
                max_zoom=opts["max_zoom"], num_steps=opts["num_steps"],
                num_repeats=opts["num_repeats"],
            )
            trajectory = sharp_camera.create_eye_trajectory(
                self._gaussians, params,
                resolution_px=(self._orig_w, self._orig_h), f_px=self._f_px,
            )
            preview_w = opts.get("preview_width", 960)
            frames = []
            total = len(trajectory)
            for i, eye_pos in enumerate(trajectory):
                if self._cancel_event.is_set():
                    break
                img = render_single(self._gaussians, self._f_px,
                                    self._orig_w, self._orig_h,
                                    eye_pos, render_width=preview_w)
                torch.cuda.synchronize()
                arr = img.cpu().numpy()
                frames.append(arr)
                self._respond("anim_frame", (arr,))
                self._respond("anim_progress", (i + 1, total))
            self._respond("anim_done", ({"n_frames": len(frames)},))
        except Exception as exc:  # noqa: BLE001
            self._respond("error", (f"动画渲染失败: {exc}",))

    def export_anim(self, opts):
        try:
            from sharp3d import video  # noqa: F401  (pins IMAGEIO_FFMPEG_EXE)
            import imageio
            path = opts["path"]
            codec = opts["codec"]
            if codec in ("av1", "libsvtav1"):
                # Resolve to the best AV1 encoder this machine has (NVENC
                # first, software fallback), honoring the frame size cap.
                fh, fw = opts["frames"][0].shape[:2] if opts.get("frames") else (0, 0)
                codec = video.resolve_av1(fw, fh)
                if codec is None:
                    raise RuntimeError(
                        "当前 ffmpeg 不支持任何 AV1 编码器"
                        "（需要 libsvtav1 / av1_nvenc / libaom-av1 之一）"
                    )
            if codec in ("libsvtav1", "av1_nvenc", "libaom-av1"):
                output_params = video.av1_output_params(codec, 18)
            else:
                output_params = ["-crf", "18", "-preset", "medium"]
                if codec == "libx265":
                    output_params += ["-tag:v", "hvc1"]
            writer = imageio.get_writer(path, fps=opts["fps"], codec=codec,
                                       quality=8, pixelformat="yuv420p",
                                       output_params=output_params)
            for f in opts["frames"]:
                writer.append_data(f)
            writer.close()
            self._respond("anim_exported", (path,))
        except Exception as exc:  # noqa: BLE001
            self._respond("error", (f"动画导出失败: {exc}",))

    # ---- Gaussian viewer: load PLY + orbit render ----------------------
    def load_ply(self, path):
        """Load a .ply gaussian file for the viewer."""
        try:
            import torch
            from sharp.utils.gaussians import load_ply
            from pathlib import Path as _P

            self._torch = torch
            self._device = torch.device("cuda")
            gaussians, metadata = load_ply(_P(path))
            # Move all gaussian tensors to CUDA (gsplat requires CUDA)
            gaussians = gaussians.to(self._device)
            self._gaussians = gaussians
            # Derive focal length and original size from metadata if available
            self._f_px = getattr(metadata, 'focal_length', None) or 1000.0
            self._orig_w = getattr(metadata, 'width', 0) or 1024
            self._orig_h = getattr(metadata, 'height', 0) or 1024
            n_g = gaussians.mean_vectors.numel() // 3
            self._respond("ply_loaded", ({
                "n_gaussians": n_g,
                "width": self._orig_w,
                "height": self._orig_h,
            },))
            self._respond("status", (f"已加载 PLY · {n_g:,} 高斯点",))
        except Exception as exc:  # noqa: BLE001
            self._respond("error", (f"PLY 加载失败: {exc}",))

    def render_orbit(self, opts):
        """Render a single orbit view from spherical coordinates."""
        try:
            import math
            torch = self._torch
            from sharp3d.render import render_single

            if self._gaussians is None:
                self._respond("error", ("请先加载 PLY 文件",))
                return

            azimuth = math.radians(opts.get("azimuth", 0.0))
            elevation = math.radians(opts.get("elevation", 0.0))
            distance = opts.get("distance", 5.0)
            render_width = opts.get("render_width", 960)

            # Spherical → Cartesian eye position
            eye_pos = torch.tensor([
                distance * math.cos(elevation) * math.sin(azimuth),
                distance * math.sin(elevation),
                distance * math.cos(elevation) * math.cos(azimuth),
            ], dtype=torch.float32, device=self._device)

            with torch.no_grad():
                img = render_single(
                    self._gaussians, self._f_px,
                    self._orig_w, self._orig_h,
                    eye_pos, render_width=render_width,
                )
            torch.cuda.synchronize()
            self._respond("orbit_frame", (img.cpu().numpy(),))
        except Exception as exc:  # noqa: BLE001
            self._respond("error", (f"渲染失败: {exc}",))


def _child_main(req_q, resp_q, cancel_event):
    """Child-process entry point. Dispatches requests to the pipeline worker."""
    try:
        _child_main_inner(req_q, resp_q, cancel_event)
    except Exception as exc:
        # Write error to log file (visible even without console)
        import traceback, sys as _sys, os as _os
        err_msg = traceback.format_exc()
        if getattr(_sys, "frozen", False):
            log_path = Path(_os.environ.get("LOCALAPPDATA", ".")) / "sharp3d" / "error.log"
        else:
            log_path = Path(__file__).resolve().parents[3] / "error.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(err_msg, encoding="utf-8")
        except Exception:
            pass
        # Tell GUI so it doesn't hang forever
        try:
            resp_q.put(("error", (f"子进程启动失败: {exc}",)))
            resp_q.put(("status", (f"错误: {exc}",)))
        except Exception:
            pass


def _child_main_inner(req_q, resp_q, cancel_event):
    """Actual child-process logic (wrapped by _child_main for error handling)."""
    # ---- 强制 UTF-8：修复中文 Windows 下 torch.compile GBK 编码错误 ----
    import os as _os
    import sys as _sys
    _os.environ["PYTHONUTF8"] = "1"

    # ---- persistent compile cache: compile once, reuse forever ----
    # Must be set before torch is imported in this process.
    if getattr(_sys, "frozen", False):
        _cache_dir = Path(_sys.executable).parent / ".cache"
    else:
        _project_root = Path(__file__).resolve().parents[3]
        _cache_dir = _project_root / ".cache"
    _os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache_dir / "inductor"))
    _os.environ.setdefault("TRITON_CACHE_DIR", str(_cache_dir / "triton"))
    _cache_dir.mkdir(parents=True, exist_ok=True)

    worker = _PipelineWorker(
        respond=lambda name, args: resp_q.put((name, args)),
        cancel_event=cancel_event,
    )
    while True:
        try:
            msg = req_q.get()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        method, kwargs = msg
        if method == "quit":
            break
        handler = getattr(worker, method, None)
        if handler is not None:
            try:
                handler(**kwargs)
            except Exception as exc:  # noqa: BLE001
                resp_q.put(("error", (f"{method} 失败: {exc}",)))
    resp_q.close()


# ===========================================================================
# GUI-process side
# ===========================================================================
class EngineProcess(QObject):
    """Bridge to the child pipeline process.

    Exposes the same signals and request methods the tabs use. Request methods
    just enqueue to the child (non-blocking); results come back via the
    response queue and are re-emitted as Qt signals on the GUI thread.
    """

    model_loading = Signal()
    model_load_progress = Signal(str, int)   # (stage name, percent 0-100)
    model_ready = Signal()
    prepared = Signal(dict)
    preview_ready = Signal(object)
    convert_progress = Signal(int, int, float, float)  # done, total, avg_fps, elapsed_s
    convert_done = Signal(dict)
    anim_frame = Signal(object)
    anim_progress = Signal(int, int)
    anim_done = Signal(dict)
    anim_exported = Signal(str)
    ply_loaded = Signal(dict)
    orbit_frame = Signal(object)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._cancel_event = ctx.Event()
        self._proc = ctx.Process(
            target=_child_main,
            args=(self._req_q, self._resp_q, self._cancel_event),
            # Must be non-daemon: torch.compile's parallel workers are child
            # processes, and daemonic processes are not allowed to have
            # children. Cleanup is still guaranteed — the child exits on
            # queue EOF when the parent dies, and stop() terminates it.
            daemon=False,
        )
        self._proc.start()

        # Poll the response queue and re-emit as Qt signals.
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(20)

    # ---- response polling ----------------------------------------------
    def _poll(self):
        try:
            while True:
                name, args = self._resp_q.get_nowait()
                sig = getattr(self, name, None)
                if sig is not None:
                    sig.emit(*args)
        except Exception:  # queue.Empty or shutdown
            pass

    # ---- request methods (non-blocking; enqueue to child) ---------------
    def preload(self):
        """Kick off model load + compile in the child right away, so the
        one-time cost overlaps with app startup instead of the first convert."""
        self._req_q.put(("preload", {}))

    def prepare(self, path, frame_idx):
        self._req_q.put(("prepare", {"path": path, "frame_idx": frame_idx}))

    def render_preview(self, ipd_mm, convergence, strength, preview_width):
        self._req_q.put(("render_preview", {
            "ipd_mm": ipd_mm, "convergence": convergence,
            "strength": strength, "preview_width": preview_width,
        }))

    def convert(self, opts):
        self._req_q.put(("convert", {"opts": opts}))

    def render_anim(self, opts):
        self._req_q.put(("render_anim", {"opts": opts}))

    def export_anim(self, opts):
        self._req_q.put(("export_anim", {"opts": opts}))

    def load_ply(self, path):
        self._req_q.put(("load_ply", {"path": path}))

    def render_orbit(self, opts):
        self._req_q.put(("render_orbit", {"opts": opts}))

    def cancel(self):
        # Cross-process cancel: set the shared event the child loop polls.
        self._cancel_event.set()

    # ---- shutdown -------------------------------------------------------
    def stop(self):
        try:
            self._poll_timer.stop()
            self._req_q.put(("quit", {}))
        except Exception:
            pass
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)
