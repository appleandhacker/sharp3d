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
import multiprocessing as mp
import os
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
        self._respond("model_load_progress", ("加载模型权重", 8))
        self._respond("status", ("正在加载 SHARP 模型权重…",))

        import torch
        from sharp.models import PredictorParams, create_predictor

        self._device = torch.device("cuda")
        state_dict = torch.hub.load_state_dict_from_url(
            "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
            progress=False, map_location="cpu",
        )
        params = PredictorParams()
        if perf_mode == "speed":
            params.monodepth.use_patch_overlap = False
            self._respond("status", ("速度模式：21 patches（精简金字塔）",))
        predictor = create_predictor(params)
        predictor.load_state_dict(state_dict)
        del state_dict  # free CPU copy immediately (~1.5GB RAM)
        predictor.eval().to(self._device)
        self._torch = torch

        # Apply channels_last memory format for Conv2d layers (lossless speedup)
        # Modern GPUs process NHWC layout more efficiently than NCHW.
        try:
            n_converted = 0
            for mod in predictor.modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    mod.weight.data = mod.weight.data.to(
                        memory_format=torch.channels_last)
                    n_converted += 1
            if n_converted:
                self._respond("status", (
                    f"channels_last 已启用（{n_converted} 个卷积层）",))
        except Exception:
            pass  # Non-critical, silently skip

        # Try ORT TensorRT acceleration for patch_encoder + image_encoder
        try:
            from sharp3d.ort_engine import (
                create_ort_patch_encoder, create_ort_image_encoder)
            spn = predictor.monodepth_model.monodepth_predictor.encoder
            use_int8 = (perf_mode == "speed")

            ort_enc = create_ort_patch_encoder(predictor, self._device,
                                               int8_enable=use_int8)
            if ort_enc is not None:
                spn.patch_encoder = ort_enc
                mode = "INT8" if use_int8 else "FP16"
                self._respond("status", (
                    f"patch_encoder: TensorRT {mode} + IO Binding",))

            ort_img = create_ort_image_encoder(predictor, self._device,
                                               int8_enable=use_int8)
            if ort_img is not None:
                spn.image_encoder = ort_img
                self._respond("status", (
                    f"image_encoder: TensorRT + IO Binding",))
        except Exception:
            pass  # Fallback to PyTorch

        # Convert remaining PyTorch weights to FP16 AFTER ORT export (export
        # needs FP32 model + FP32 dummy input). ORT encoders have no PyTorch
        # parameters so .half() is a no-op on them. Saves ~1.5GB VRAM.
        predictor.half()
        self._respond("status", ("模型权重 FP16（节省 ~1.5GB 显存）",))

        self._pipeline = predictor

        # ── Pre-build TensorRT engines (before torch.compile) ─────────
        # Running the ORT encoders once triggers TRT engine building,
        # which is cached for subsequent runs. Doing this BEFORE
        # torch.compile separates the two slow operations and gives
        # the user visible progress instead of a single long wait.
        try:
            spn = predictor.monodepth_model.monodepth_predictor.encoder
            if hasattr(spn.patch_encoder, '_session'):
                n_patches = 35 if params.monodepth.use_patch_overlap else 21
                self._respond("model_load_progress", ("构建 TensorRT 引擎", 35))
                self._respond("status", (
                    f"正在构建 TensorRT 引擎（{n_patches} patches，首次约30秒）…",))
                dummy_patches = torch.zeros(
                    n_patches, 3, 384, 384, device=self._device)
                with torch.no_grad():
                    spn.patch_encoder(dummy_patches)
                torch.cuda.synchronize()
                del dummy_patches
            if hasattr(spn.image_encoder, '_session'):
                self._respond("status", ("正在构建 image_encoder TensorRT 引擎…",))
                dummy_img_enc = torch.zeros(1, 3, 384, 384, device=self._device)
                with torch.no_grad():
                    spn.image_encoder(dummy_img_enc)
                torch.cuda.synchronize()
                del dummy_img_enc
        except Exception:
            pass  # Non-critical; engines will build on first real use

        self._respond("model_load_progress", ("编译预测器", 55))
        self._respond("status", ("正在编译预测器 (torch.compile)…",))
        torch._dynamo.config.capture_scalar_outputs = True
        # ORT encoders are @torch.compiler.disable'd → graph breaks.
        # CUDA Graph capture hangs on these breaks (same issue as FP8 testing).
        torch._inductor.config.triton.cudagraphs = False
        # max-autotune spawns one worker per CPU core by default (32 on this
        # machine); each loads the model for benchmarking → memory explosion.
        # Auto-size the pool from available RAM (~3GB/worker, 4GB OS headroom)
        # so compilation stays fast AND memory stays flat.
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / 1e9
            n_workers = max(1, min(os.cpu_count() or 1, int((avail_gb - 4) // 3)))
        except Exception:
            n_workers = 4  # safe fallback
        torch._inductor.config.compile_threads = n_workers
        # compile_threads>1 spawns worker subprocesses; the default 'subprocess'
        # start method uses pass_fds which crashes on Windows. Force 'spawn'.
        os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")
        self._respond("status", (f"编译线程: {n_workers}（按可用内存自动分配）",))
        # Coordinate-descent kernel tuning is the most memory/time-intensive
        # autotune phase for only marginal runtime gain — skip it.
        torch._inductor.config.coordinate_descent_tuning = False
        self._compiled = torch.compile(predictor, mode="max-autotune", dynamic=False)

        from sharp3d.unproject import INTERNAL_SHAPE
        dummy_img = torch.zeros(1, 3, *INTERNAL_SHAPE, device=self._device)
        dummy_df = torch.tensor([1.0], device=self._device, dtype=torch.float32)
        self._respond("model_load_progress", ("预热推理", 72))
        self._respond("status", ("正在预热推理（编译内核，首次约1分钟）…",))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g_ndc = self._compiled(dummy_img, dummy_df)
        torch.cuda.synchronize()

        # Trigger gsplat's one-time CUDA JIT here as well (it compiles its
        # rasterization kernels on first use), so the first real frame
        # renders at full speed instead of stalling ~15s.
        self._respond("model_load_progress", ("编译渲染内核", 90))
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

        self._respond("model_load_progress", ("就绪", 100))
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
            self._ensure_pipeline(opts.get("perf_mode", "quality"))
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
        sbs, (sw, sh) = render_sbs(g, f_px, w, h, ipd=ipd_scene,
                                   convergence=conv, render_width=render_w)
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

    def _convert_video(self, path, out, opts, ipd_scene, conv, method,
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

        # ── Frame-rate conversion: map output frames → source frames ──
        # out_per_src[i] = how many output frames source frame i produces.
        # out_fps < source → decimation (some source frames skipped → fewer
        # frames converted → faster). out_fps > source → duplication.
        src_fps = reader.fps or out_fps
        if out_fps and src_fps and abs(out_fps - src_fps) > 1e-6:
            out_count = max(1, int(round(n * out_fps / src_fps)))
            out_per_src = [0] * n
            for j in range(out_count):
                si = min(n - 1, int(round(j * src_fps / out_fps)))
                out_per_src[si] += 1
        else:
            out_count = n
            out_per_src = [1] * n

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
        # Queue of 4 frames balances decode prefetch against RAM pressure
        # (4 × 4K frame ≈ 100MB). Larger queues risk pushing the system
        # into swap when shared GPU memory already consumes most RAM.
        import queue as _queue
        import threading

        frame_q: _queue.Queue = _queue.Queue(maxsize=4)

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

        def _prepare(frm):
            return prepare_input(frm, f_px, self._device, async_upload=False)

        def _used_frames():
            """Yield (frame_np, n_outputs) for each source frame kept in the
            output (out_per_src[i] > 0); skipped frames are drained but not
            yielded, so decimated frames cost nothing beyond decode."""
            si = 0
            while True:
                frm = frame_q.get()
                if frm is None:
                    return
                cnt = out_per_src[si] if si < len(out_per_src) else 0
                si += 1
                if cnt > 0:
                    yield frm, cnt

        # ── Main conversion loop ────────────────────────────────────────
        # Wall-clock timing: elapsed and avg_fps are computed from the real
        # start time, NOT from instantaneous per-frame dt (which caused the
        # timer to jump back and forth as frame times varied).
        out_written = 0
        t_start = time.time()
        n_done = 0
        used = _used_frames()
        first = next(used, None)
        if first is not None and not self._cancel_event.is_set():
            frm, cnt = first
            prepared = _prepare(frm)
            del frm

            while True:
                img_r, df, ir, (w, h) = prepared

                # ── Predict + unproject (GPU-bound, ~500ms) ─────────
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    g_ndc = self._compiled(img_r, df)
                g = fast_unproject(g_ndc, torch.eye(4, device=self._device), ir,
                                   INTERNAL_SHAPE, decompose_method=method)

                # ── Render + pack (GPU) ─────────────────────────────
                sbs, _ = render_sbs(g, f_px, w, h,
                                    ipd=ipd_scene, convergence=conv,
                                    render_width=render_w)
                packed = pack_stereo(fmt, sbs)
                torch.cuda.synchronize()

                # ── Encode (write this frame `cnt` times) ───────────
                sbs_np = packed.cpu().numpy()
                for _ in range(cnt):
                    if hdr_out:
                        writer.write_frame(sbs_np)
                    else:
                        writer.append_frame(sbs_np)
                    out_written += 1

                # Free GPU tensors immediately after encode to keep VRAM flat
                del g, g_ndc, sbs, packed, img_r, prepared, sbs_np

                n_done += 1
                elapsed = time.time() - t_start
                avg_fps = n_done / elapsed if elapsed > 0 else 0.0
                self._respond("convert_progress",
                              (out_written, out_count, avg_fps, elapsed))

                # Periodic cleanup every 30 frames: gc.collect() prevents RAM
                # accumulation; empty_cache() returns fragmented VRAM blocks to
                # CUDA so the allocator doesn't slowly creep into shared memory.
                if n_done % 30 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                if self._cancel_event.is_set():
                    break

                # ── Fetch + prepare next frame (overlaps with encode I/O
                #    settling; the GPU is free here so the H2D copy and
                #    F.interpolate resize run without contention) ─────
                nxt = next(used, None)
                if nxt is None:
                    break
                nxt_frm, cnt = nxt
                prepared = _prepare(nxt_frm)
                del nxt_frm

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


def _child_main(req_q, resp_q, cancel_event):
    """Child-process entry point. Dispatches requests to the pipeline worker."""
    # ---- persistent compile cache: compile once, reuse forever ----
    # Must be set before torch is imported in this process.
    import os as _os
    _project_root = Path(__file__).resolve().parents[3]  # sharp3d/src/sharp3d/gui -> sharp3d/
    _cache_dir = _project_root / ".cache"
    _os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache_dir / "inductor"))
    _os.environ.setdefault("TRITON_CACHE_DIR", str(_cache_dir / "triton"))
    # Default 'subprocess' worker start uses pass_fds → crashes on Windows.
    # 'spawn' enables safe multithreaded compilation (see compile_threads).
    _os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")
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
