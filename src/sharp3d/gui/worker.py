"""GPU worker engine for the sharp3d GUI.

A persistent QObject living on a dedicated QThread. The heavy torch/SHARP
stack is imported lazily on first use (inside the worker thread) so the GUI
starts instantly. Requests are queued via signals and executed sequentially,
which keeps GPU operations serialized and thread-safe.

Design for responsive stereo tuning:
  - prepare(): expensive predict + unproject → caches world-space gaussians
  - render_preview(): cheap re-render from the cache with new IPD/convergence
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot


class Engine(QObject):
    """Persistent GPU pipeline worker."""

    # --- signals back to the GUI ---
    model_loading = Signal()
    model_ready = Signal()
    prepared = Signal(dict)            # {width, height, n_gaussians}
    preview_ready = Signal(object)     # (H, W, 3) uint8 numpy SBS
    convert_progress = Signal(int, int, float)  # frame, total, fps
    convert_done = Signal(dict)
    anim_frame = Signal(object)        # (H, W, 3) uint8 numpy frame
    anim_progress = Signal(int, int)
    anim_done = Signal(dict)
    error = Signal(str)
    status = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None       # lazy-loaded
        self._compiled = None
        self._device = None
        self._gaussians = None      # cached world-space gaussians
        self._f_px = 1.0
        self._orig_w = 0
        self._orig_h = 0
        self._cancel = False

    # ------------------------------------------------------------------
    # Lazy pipeline loading (runs inside worker thread)
    # ------------------------------------------------------------------
    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        self.model_loading.emit()
        self.status.emit("正在加载 SHARP 模型并编译…")

        import torch
        from sharp.models import PredictorParams, create_predictor

        self._device = torch.device("cuda")
        state_dict = torch.hub.load_state_dict_from_url(
            "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
            progress=False, map_location="cpu",
        )
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval().to(self._device)
        self._compiled = torch.compile(predictor, mode="default")
        self._predictor_params = PredictorParams()
        self._torch = torch
        self._pipeline = predictor

        # warmup to trigger compilation
        from sharp3d.unproject import INTERNAL_SHAPE
        dummy_img = torch.zeros(1, 3, *INTERNAL_SHAPE, device=self._device)
        dummy_df = torch.tensor([1.0], device=self._device, dtype=torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            _ = self._compiled(dummy_img, dummy_df)
        torch.cuda.synchronize()
        del dummy_img, dummy_df

        self.model_ready.emit()
        self.status.emit("模型就绪")

    # ------------------------------------------------------------------
    # Prepare: predict + unproject a frame → cache gaussians
    # ------------------------------------------------------------------
    @Slot(str, int)
    def prepare(self, path: str, frame_idx: int) -> None:
        try:
            self._ensure_pipeline()
            torch = self._torch
            from sharp.utils import io as sharp_io
            from sharp3d.unproject import prepare_input, fast_unproject, INTERNAL_SHAPE

            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
            p = Path(path)

            if p.suffix.lower() in video_exts:
                # FrameReader tone-maps HDR sources to SDR for the model
                from sharp3d.hdr import FrameReader
                reader = FrameReader(p)
                frame = reader.read_frame(frame_idx)
                f_px = frame.shape[1] * 1.2  # estimate (~60 deg FOV)
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
            self.prepared.emit({"width": w, "height": h, "n_gaussians": n_g, "f_px": f_px})
            self.status.emit(f"已重建 3D 场景 · {w}×{h}")
            del g_ndc, img_r
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"准备失败: {exc}")

    # ------------------------------------------------------------------
    # Render preview from cached gaussians (fast, slider-responsive)
    # ------------------------------------------------------------------
    @Slot(float, float, float, int)
    def render_preview(self, ipd_mm: float, convergence: float, strength: float,
                       preview_width: int) -> None:
        if self._gaussians is None:
            return
        try:
            torch = self._torch
            from sharp3d.render import render_sbs

            ipd_scene = (ipd_mm / 1000.0) * strength
            conv = None if convergence <= 0 else convergence

            with torch.no_grad():
                sbs, _ = render_sbs(
                    self._gaussians, self._f_px, self._orig_w, self._orig_h,
                    ipd=ipd_scene, convergence=conv, render_width=preview_width,
                )
            torch.cuda.synchronize()
            self.preview_ready.emit(sbs.cpu().numpy())
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"预览渲染失败: {exc}")

    # ------------------------------------------------------------------
    # Full video / image conversion
    # ------------------------------------------------------------------
    @Slot(dict)
    def convert(self, opts: dict) -> None:
        try:
            self._ensure_pipeline()
            torch = self._torch
            from sharp.utils import io as sharp_io
            from sharp3d.unproject import prepare_input, fast_unproject, INTERNAL_SHAPE
            from sharp3d.render import render_sbs

            self._cancel = False
            path = Path(opts["input"])
            out = Path(opts["output"])
            ipd_scene = (opts["ipd_mm"] / 1000.0) * opts["strength"]
            conv = None if opts["convergence"] <= 0 else opts["convergence"]
            method = opts.get("decompose", "analytical")

            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
            is_video = path.suffix.lower() in video_exts

            if is_video:
                self._convert_video(path, out, opts, ipd_scene, conv, method,
                                    prepare_input, fast_unproject, render_sbs,
                                    INTERNAL_SHAPE, torch)
            else:
                self._convert_image(path, out, opts, ipd_scene, conv, method,
                                    prepare_input, fast_unproject, render_sbs,
                                    INTERNAL_SHAPE, torch, sharp_io)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"转换失败: {exc}")

    def _convert_image(self, path, out, opts, ipd_scene, conv, method,
                       prepare_input, fast_unproject, render_sbs,
                       INTERNAL_SHAPE, torch, sharp_io) -> None:
        from PIL import Image

        image_np, _, f_px = sharp_io.load_rgb(path)
        h, w = image_np.shape[:2]
        img_r, df, ir, _ = prepare_input(image_np, f_px, self._device)

        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g_ndc = self._compiled(img_r, df)
        g = fast_unproject(g_ndc, torch.eye(4, device=self._device), ir,
                           INTERNAL_SHAPE, decompose_method=method)
        sbs, (sw, sh) = render_sbs(g, f_px, w, h, ipd=ipd_scene, convergence=conv)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        Image.fromarray(sbs.cpu().numpy()).save(out)

        # optional depth map
        if opts.get("depth"):
            from sharp3d.render import render_depth_map
            depth = render_depth_map(g, f_px, w, h)
            depth_path = out.with_stem(out.stem + "_depth")
            Image.fromarray(depth.cpu().numpy()).save(depth_path)

        # optional PLY export (world-space gaussians)
        if opts.get("ply"):
            from sharp.utils.gaussians import save_ply
            ply_path = out.with_suffix(".ply")
            save_ply(g, f_px, (h, w), ply_path)

        del g, g_ndc, sbs
        torch.cuda.empty_cache()
        gc.collect()

        self.convert_progress.emit(1, 1, 1.0 / elapsed)
        self.convert_done.emit({
            "output": str(out), "elapsed": elapsed, "fps": 1.0 / elapsed,
            "n_frames": 1, "size": (sw * 2, sh),
        })

    def _convert_video(self, path, out, opts, ipd_scene, conv, method,
                       prepare_input, fast_unproject, render_sbs,
                       INTERNAL_SHAPE, torch) -> None:
        from sharp3d.hdr import FrameReader, Hdr10Writer
        from sharp3d.video import VideoWriter

        reader = FrameReader(path)  # tone-maps HDR input to SDR for the model
        n = reader.n_frames
        f_px = reader.width * 1.2  # estimate (~60 deg FOV)

        hdr_out = opts.get("hdr_output", False)
        if hdr_out:
            writer = Hdr10Writer(
                out, width=reader.width * 2, height=reader.height,
                fps=reader.fps, codec=opts.get("codec", "h265"),
                crf=opts.get("crf", 18),
            )
        else:
            writer = VideoWriter(
                out, fps=reader.fps, width=reader.width * 2, height=reader.height,
                codec=opts.get("codec", "h264"), crf=opts.get("crf", 18),
            )

        frame_times = []
        for i, frame in enumerate(reader.stream_frames()):
            if self._cancel:
                break
            img_r, df, ir, (w, h) = prepare_input(frame, f_px, self._device)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                g_ndc = self._compiled(img_r, df)
            g = fast_unproject(g_ndc, torch.eye(4, device=self._device), ir,
                               INTERNAL_SHAPE, decompose_method=method)
            sbs, _ = render_sbs(g, f_px, w, h, ipd=ipd_scene, convergence=conv)
            torch.cuda.synchronize()
            dt = time.time() - t0
            frame_times.append(dt)
            sbs_np = sbs.cpu().numpy()
            if hdr_out:
                writer.write_frame(sbs_np)
            else:
                writer.append_frame(sbs_np)
            self.convert_progress.emit(i + 1, n, 1.0 / dt)
            del g, g_ndc, sbs, img_r, frame
            torch.cuda.empty_cache()

        source = path if reader.has_audio else None
        if hdr_out:
            writer.close(audio_source=source)
        else:
            writer.close(source_video=source)

        avg = float(np.mean(frame_times)) if frame_times else 0.0
        self.convert_done.emit({
            "output": str(out), "elapsed": sum(frame_times), "fps": 1.0 / avg if avg else 0.0,
            "n_frames": len(frame_times), "size": (reader.width * 2, reader.height),
            "cancelled": self._cancel, "hdr": hdr_out,
        })

    @Slot()
    def cancel(self) -> None:
        self._cancel = True

    # ------------------------------------------------------------------
    # 2.5D parallax animation
    # ------------------------------------------------------------------
    @Slot(dict)
    def render_anim(self, opts: dict) -> None:
        if self._gaussians is None:
            self.error.emit("请先加载一张图片")
            return
        try:
            torch = self._torch
            from sharp.utils import camera as sharp_camera
            from sharp3d.render import render_single

            self._cancel = False
            params = sharp_camera.TrajectoryParams(
                type=opts["type"],
                max_disparity=opts["max_disparity"],
                max_zoom=opts["max_zoom"],
                num_steps=opts["num_steps"],
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
                if self._cancel:
                    break
                img = render_single(
                    self._gaussians, self._f_px, self._orig_w, self._orig_h,
                    eye_pos, render_width=preview_w,
                )
                torch.cuda.synchronize()
                arr = img.cpu().numpy()
                frames.append(arr)
                self.anim_frame.emit(arr)
                self.anim_progress.emit(i + 1, total)

            self.anim_done.emit({"n_frames": len(frames), "frames": frames})
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"动画渲染失败: {exc}")


class EngineThread:
    """Owns the worker QThread and exposes the engine + its signals."""

    def __init__(self) -> None:
        self.thread = QThread()
        self.engine = Engine()
        self.engine.moveToThread(self.thread)
        self.thread.start()

    def stop(self) -> None:
        self.thread.quit()
        self.thread.wait(3000)
