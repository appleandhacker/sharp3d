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

from __future__ import annotations

import gc
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal


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

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        self._respond("model_loading", ())
        self._respond("status", ("正在加载 SHARP 模型权重…",))

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
        self._torch = torch
        self._pipeline = predictor

        self._respond("status", ("正在编译预测器 (torch.compile)…",))
        torch._dynamo.config.capture_scalar_outputs = True
        self._compiled = torch.compile(predictor, mode="max-autotune", dynamic=False)

        from sharp3d.unproject import INTERNAL_SHAPE
        dummy_img = torch.zeros(1, 3, *INTERNAL_SHAPE, device=self._device)
        dummy_df = torch.tensor([1.0], device=self._device, dtype=torch.float32)
        self._respond("status", ("正在预热推理（首次需编译内核，约1分钟）…",))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g_ndc = self._compiled(dummy_img, dummy_df)
        torch.cuda.synchronize()

        # Trigger gsplat's one-time CUDA JIT here as well (it compiles its
        # rasterization kernels on first use), so the first real frame
        # renders at full speed instead of stalling ~15s.
        self._respond("status", ("正在编译渲染内核 (gsplat)…",))
        try:
            from sharp3d.unproject import fast_unproject
            from sharp3d.render import render_sbs
            f = INTERNAL_SHAPE[1] * 1.2
            ir = torch.tensor([
                [f, 0, (INTERNAL_SHAPE[1] - 1) / 2.0, 0],
                [0, f, (INTERNAL_SHAPE[0] - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32, device=self._device)
            g = fast_unproject(g_ndc, torch.eye(4, device=self._device), ir,
                               INTERNAL_SHAPE, decompose_method="analytical")
            render_sbs(g, f, INTERNAL_SHAPE[1], INTERNAL_SHAPE[0],
                       ipd=0.063, render_width=320)
            torch.cuda.synchronize()
            del g, ir
        except Exception:  # noqa: BLE001
            pass  # warmup only; worst case the first real render JITs instead
        del dummy_img, dummy_df, g_ndc
        torch.cuda.empty_cache()

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
            from sharp3d.render import render_sbs
            ipd_scene = (ipd_mm / 1000.0) * strength
            conv = None if convergence <= 0 else convergence
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
            self._ensure_pipeline()
            torch = self._torch
            from sharp.utils import io as sharp_io
            from sharp3d.unproject import prepare_input, fast_unproject, INTERNAL_SHAPE
            from sharp3d.render import render_sbs

            self._cancel_event.clear()
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
            self._respond("error", (f"转换失败: {exc}",))

    def _convert_image(self, path, out, opts, ipd_scene, conv, method,
                       prepare_input, fast_unproject, render_sbs,
                       INTERNAL_SHAPE, torch, sharp_io):
        from PIL import Image
        from sharp3d.formats import output_size, pack as pack_stereo

        fmt = opts.get("format", "full_sbs")
        image_np, _, f_px = sharp_io.load_rgb(path)
        h, w = image_np.shape[:2]
        img_r, df, ir, _ = prepare_input(image_np, f_px, self._device)

        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g_ndc = self._compiled(img_r, df)
        g = fast_unproject(g_ndc, torch.eye(4, device=self._device), ir,
                           INTERNAL_SHAPE, decompose_method=method)
        sbs, (sw, sh) = render_sbs(g, f_px, w, h, ipd=ipd_scene, convergence=conv)
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

        self._respond("convert_progress", (1, 1, 1.0 / elapsed))
        self._respond("convert_done", ({
            "output": str(out), "elapsed": elapsed, "fps": 1.0 / elapsed,
            "n_frames": 1, "size": output_size(fmt, sw, sh),
        },))

    def _convert_video(self, path, out, opts, ipd_scene, conv, method,
                       prepare_input, fast_unproject, render_sbs,
                       INTERNAL_SHAPE, torch):
        from sharp3d.hdr import FrameReader, Hdr10Writer
        from sharp3d.video import VideoWriter, resolve_av1
        from sharp3d.formats import output_size, pack as pack_stereo

        reader = FrameReader(path)
        n = reader.n_frames
        f_px = reader.width * 1.2

        fmt = opts.get("format", "full_sbs")
        out_w, out_h = output_size(fmt, reader.width, reader.height)

        # Tell the user when AV1 silently falls back to CPU encoding because
        # the packed output is too large for the GPU encoder (NVENC caps at 8192).
        if opts.get("codec") == "av1" and not opts.get("hdr_output", False):
            enc = resolve_av1(out_w, out_h)
            if enc and enc != "av1_nvenc":
                self._respond("status", (
                    f"输出 {out_w}×{out_h} 超过GPU编码上限，"
                    f"AV1 改用CPU编码 ({enc})",))

        hdr_out = opts.get("hdr_output", False)
        if hdr_out:
            writer = Hdr10Writer(out, width=out_w, height=out_h,
                                 fps=reader.fps, codec=opts.get("codec", "h265"),
                                 crf=opts.get("crf", 18))
        else:
            writer = VideoWriter(out, fps=reader.fps, width=out_w, height=out_h,
                                 codec=opts.get("codec", "h264"),
                                 crf=opts.get("crf", 18))

        frame_times = []

        # Prefetch: a background thread decodes frames from the ffmpeg pipe
        # while the GPU renders, so decode time overlaps rendering instead of
        # serializing with it. (Plain threads are safe here — this runs in
        # the Qt-free child process.)
        import queue as _queue
        import threading

        frame_q: _queue.Queue = _queue.Queue(maxsize=3)

        def _decode():
            try:
                for frm in reader.stream_frames():
                    if self._cancel_event.is_set():
                        break
                    frame_q.put(frm)
            finally:
                frame_q.put(None)

        decoder = threading.Thread(target=_decode, daemon=True)
        decoder.start()

        # Pipeline the host->device transfer too: prepare frame N+1 on a side
        # stream (pinned non-blocking upload + resize) while frame N is being
        # predicted/rendered on the main stream. Copy and compute use separate
        # DMA/compute engines, so the transfer overlaps GPU work instead of
        # serializing with it.
        side_stream = torch.cuda.Stream()
        main_stream = torch.cuda.current_stream()

        def _prepare_async(frm):
            with torch.cuda.stream(side_stream):
                prepared = prepare_input(frm, f_px, self._device,
                                         async_upload=True)
                upload_done = side_stream.record_event()
            # Wait for the copy on the CPU: the pinned host buffer is freed
            # when prepare_input returns, so it must not be reused (by the
            # next frame's pin_memory) while the async copy still reads it.
            # This stalls only the CPU ~10ms — the GPU keeps rendering the
            # previous frame; the side stream provides the real overlap.
            upload_done.synchronize()
            return prepared, upload_done

        i = 0
        first = frame_q.get()
        if first is not None and not self._cancel_event.is_set():
            prepared, upload_done = _prepare_async(first)
            del first
            while True:
                # Fetch the next frame and kick off its prepare now, so its
                # upload overlaps this frame's GPU pass.
                nxt = frame_q.get()
                nxt_prepared = None
                if nxt is not None and not self._cancel_event.is_set():
                    nxt_prepared = _prepare_async(nxt)
                del nxt

                main_stream.wait_event(upload_done)
                img_r, df, ir, (w, h) = prepared
                t0 = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    g_ndc = self._compiled(img_r, df)
                g = fast_unproject(g_ndc, torch.eye(4, device=self._device), ir,
                                   INTERNAL_SHAPE, decompose_method=method)
                sbs, _ = render_sbs(g, f_px, w, h, ipd=ipd_scene, convergence=conv)
                packed = pack_stereo(fmt, sbs)
                torch.cuda.synchronize()
                dt = time.time() - t0
                frame_times.append(dt)
                sbs_np = packed.cpu().numpy()
                if hdr_out:
                    writer.write_frame(sbs_np)
                else:
                    writer.append_frame(sbs_np)
                self._respond("convert_progress", (i + 1, n, 1.0 / dt))
                i += 1
                del g, g_ndc, sbs, packed, img_r, prepared
                # Deliberately NO per-frame torch.cuda.empty_cache(): it forces
                # a device sync plus allocator churn on every frame. Shapes are
                # constant frame-to-frame, so the caching allocator reuses its
                # blocks and VRAM stays flat.

                if nxt_prepared is None:
                    break
                prepared, upload_done = nxt_prepared

        # Unblock the producer if it is parked on a full queue, then join.
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

        avg = float(np.mean(frame_times)) if frame_times else 0.0
        self._respond("convert_done", ({
            "output": str(out), "elapsed": sum(frame_times),
            "fps": 1.0 / avg if avg else 0.0,
            "n_frames": len(frame_times), "size": (out_w, out_h),
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


def _child_main(req_q, resp_q, cancel_event):
    """Child-process entry point. Dispatches requests to the pipeline worker."""
    # ---- persistent compile cache: compile once, reuse forever ----
    # Must be set before torch is imported in this process.
    import os as _os
    _project_root = Path(__file__).resolve().parents[3]  # sharp3d/src/sharp3d/gui -> sharp3d/
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
    model_ready = Signal()
    prepared = Signal(dict)
    preview_ready = Signal(object)
    convert_progress = Signal(int, int, float)
    convert_done = Signal(dict)
    anim_frame = Signal(object)
    anim_progress = Signal(int, int)
    anim_done = Signal(dict)
    anim_exported = Signal(str)
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
            daemon=True,
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
