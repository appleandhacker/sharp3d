"""Full pipeline: image/video → SHARP predict → unproject → SBS render → output.

Orchestrates all modules into a single coherent pipeline.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from sharp.utils import io as sharp_io

from .predict import SharpPredictor
from .unproject import prepare_input, fast_unproject, INTERNAL_SHAPE
from .render import render_sbs, render_depth_map
from .video import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


class Sharp3DPipeline:
    """End-to-end 2D→3D conversion pipeline."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        use_compile: bool = True,
        use_fp16: bool = True,
        decompose_method: str = "analytical",
        ipd: float = 0.063,
    ):
        self.device = device
        self.decompose_method = decompose_method
        self.ipd = ipd

        # Load model (torch.compile handled internally with auto-fallback)
        import os
        if not use_compile:
            os.environ["SHARP3D_NO_COMPILE"] = "1"
        self.predictor = SharpPredictor(device=device, fp16=use_fp16)
        # SharpPredictor already warms up (predict + gsplat kernel) in
        # __init__, so there is nothing left to trigger lazily.

    def process_image(
        self,
        input_path: str | Path,
        output_path: str | Path,
        output_depth: bool = False,
        format: str = "full_sbs",
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """Process a single image → stereoscopic output.

        Args:
            input_path: Path to input image.
            output_path: Path for stereo output image.
            output_depth: Also save depth map visualization.
            format: Stereo packing format (see sharp3d.formats).
            progress_callback: Optional callback for status updates.

        Returns:
            dict with timing info and output paths.
        """
        from .formats import output_size, pack as pack_stereo

        input_path = Path(input_path)
        output_path = Path(output_path)

        if progress_callback:
            progress_callback("Loading image...")

        image_np, _, f_px = sharp_io.load_rgb(input_path)
        h, w = image_np.shape[:2]

        if progress_callback:
            progress_callback("Preparing input...")

        img_resized, df, intrinsics_resized, (orig_w, orig_h) = prepare_input(
            image_np, f_px, self.device
        )


        if progress_callback:
            progress_callback("Predicting 3D Gaussians...")

        torch.cuda.synchronize()
        t0 = time.time()

        g_ndc = self.predictor.predict(img_resized, df)

        if progress_callback:
            progress_callback("Rendering SBS...")

        if output_depth:
            # Depth rendering needs world-space gaussians.
            g_world = fast_unproject(
                g_ndc,
                torch.eye(4, device=self.device),
                intrinsics_resized,
                INTERNAL_SHAPE,
                decompose_method=self.decompose_method,
            )
            sbs_img, (sw, sh) = render_sbs(
                g_world, f_px, orig_w, orig_h, ipd=self.ipd
            )
        else:
            # Fast path: fold the unprojection into the view matrices and
            # render the NDC gaussians directly (skips the per-frame
            # covariance compose→transform→eigendecompose round-trip).
            from sharp.utils.gaussians import get_unprojection_matrix
            U = get_unprojection_matrix(
                torch.eye(4, device=self.device),
                intrinsics_resized, INTERNAL_SHAPE,
            )
            sbs_img, (sw, sh) = render_sbs(
                g_ndc, f_px, orig_w, orig_h, ipd=self.ipd, ndc_transform=U
            )

        torch.cuda.synchronize()
        elapsed = time.time() - t0

        # Save output
        packed = pack_stereo(format, sbs_img)
        sbs_np = packed.cpu().numpy()
        from sharp3d.imgio import save_rgb
        save_rgb(output_path, sbs_np)

        result = {
            "elapsed": elapsed,
            "fps": 1.0 / elapsed if elapsed > 1e-9 else 0.0,
            "output_path": str(output_path),
            "output_size": output_size(format, sw, sh),
        }

        # Optional depth map
        if output_depth:
            depth_path = output_path.with_stem(output_path.stem + "_depth")
            depth_img = render_depth_map(g_world, f_px, orig_w, orig_h)
            from sharp3d.imgio import save_rgb
            save_rgb(depth_path, depth_img.cpu().numpy())
            result["depth_path"] = str(depth_path)

        # Cleanup
        if output_depth:
            del g_world
        del g_ndc, sbs_img
        torch.cuda.empty_cache()
        gc.collect()

        return result

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        codec: str = "h264",
        crf: int = 26,
        format: str = "full_sbs",
        hdr_output: bool | None = None,
        keyframe_interval: int = 1,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> dict:
        """Process video → stereoscopic video with audio.

        Args:
            input_path: Path to input video.
            output_path: Path for stereo output video.
            codec: "h264", "h265", or "av1".
            crf: Quality (lower = better).
            format: Stereo packing format (see sharp3d.formats).
            hdr_output: True = force HDR10 output, False = force SDR,
                        None = auto (HDR10 if the input is HDR).
            keyframe_interval: Full SHARP prediction every Nth frame;
                in-between frames reuse keyframe geometry with refreshed
                colors (see VideoConversionEngine). 1 = every frame.
            progress_callback: (frame_idx, total_frames, fps) callback.

        Returns:
            dict with timing info and output path.
        """
        from .formats import output_size, pack as pack_stereo
        from .hdr import FrameReader, Hdr10Writer, probe_video

        input_path = Path(input_path)
        output_path = Path(output_path)

        info = probe_video(input_path)
        reader = FrameReader(input_path, info)  # tone-maps HDR input to SDR
        n_frames = reader.n_frames
        vid_fps = reader.fps
        f_px = reader.width * 1.2  # ~60° FOV estimate

        # The writer must be opened at the size the renderer actually emits,
        # which is the *screen* resolution (halved for sources taller than
        # 3000px), not the source resolution. Using reader.width/height here
        # desynchronized the writer from the frames for any 4K+ input.
        from .render import _get_screen_resolution
        sw, sh = _get_screen_resolution(reader.width, reader.height)
        out_w, out_h = output_size(format, sw, sh)

        # HDR output: explicit flag, or auto-match the input's HDR status
        is_hdr = info["is_hdr"]
        want_hdr = is_hdr if hdr_output is None else hdr_output
        if want_hdr and codec == "h264":
            codec = "h265"  # H.264 cannot carry HDR10

        if want_hdr:
            writer = Hdr10Writer(
                output_path, fps=vid_fps, width=out_w,
                height=out_h, codec=codec, crf=crf,
            )
        else:
            writer = VideoWriter(
                output_path, fps=vid_fps,
                width=out_w, height=out_h,
                codec=codec, crf=crf,
            )

        frame_times = []
        total_start = time.time()

        # Prefetch: decode frames in a background thread so the ffmpeg pipe
        # read overlaps GPU rendering instead of serializing with it.
        import queue as _queue
        import threading

        frame_q: _queue.Queue = _queue.Queue(maxsize=3)

        def _decode():
            try:
                for frm in reader.stream_frames():
                    frame_q.put(frm)
            finally:
                frame_q.put(None)

        decoder = threading.Thread(target=_decode, daemon=True)
        decoder.start()

        ok = False
        try:
            # Temporal stabilization + keyframe reuse via the shared engine
            # (same code path as the GUI worker).
            from .conversion import VideoConversionEngine
            engine = VideoConversionEngine(
                predict_fn=self.predictor,
                device=self.device,
                f_px=f_px,
                fmt=format,
                ipd=self.ipd,
                decompose_method=self.decompose_method,
                stabilize_mode="adaptive",
                keyframe_interval=keyframe_interval,
            )

            # Pipeline the host->device transfer: prepare frame N+1 on a side
            # stream while frame N renders on the main stream (copy and compute
            # use separate engines, so the upload overlaps GPU work).
            side_stream = torch.cuda.Stream()
            main_stream = torch.cuda.current_stream()

            def _prepare_async(frm):
                with torch.cuda.stream(side_stream):
                    prepared = prepare_input(frm, f_px, self.device,
                                             async_upload=True)
                    upload_done = side_stream.record_event()
                # No CPU-side event wait here. PyTorch's CachingHostAllocator
                # records an event on the copy stream when the pinned block is
                # freed, so a later pin_memory() can only reuse it after the copy
                # finished — buffer safety does not need a manual synchronize.
                # Waiting here would stall the CPU *before* the current frame's
                # GPU work is even submitted (the caller prepares frame N+1 first),
                # leaving the GPU idle every frame and defeating the prefetch.
                return prepared, upload_done

            i = 0

            # Optional per-stage profiling (SHARP3D_PROFILE=1). The engine marks
            # predict/stabilize/render internally; we add the D2H+write wall time.
            from . import profiling
            prof = profiling.get_timer() if profiling.ENABLED else None

            first = frame_q.get()
            if first is not None:
                prepared, upload_done = _prepare_async(first)
                del first
                while True:
                    nxt = frame_q.get()
                    nxt_prepared = _prepare_async(nxt) if nxt is not None else None
                    del nxt

                    main_stream.wait_event(upload_done)
                    img_resized, df, intrinsics_resized, (orig_w, orig_h) = prepared

                    t0 = time.time()

                    packed = engine.process_frame(
                        img_resized, df, intrinsics_resized, (orig_w, orig_h),
                        download=False,
                    )
                    if prof:
                        _t_d2h = time.time()
                    # packed.cpu() is a blocking copy — it implicitly waits for
                    # the render stream, so no torch.cuda.synchronize() is needed
                    # (a device-wide sync would also stall on the next frame's
                    # side-stream upload and break the overlap).
                    sbs_np = packed.cpu().numpy()
                    dt = time.time() - t0
                    frame_times.append(dt)
                    if want_hdr:
                        writer.write_frame(sbs_np)
                    else:
                        writer.append_frame(sbs_np)
                    if prof:
                        prof.frame_end((time.time() - _t_d2h) * 1000.0)

                    if progress_callback:
                        progress_callback(i, n_frames, 1.0 / dt if dt > 1e-6 else 0.0)
                    i += 1

                    del packed, img_resized, prepared
                    # No per-frame empty_cache(): constant shapes mean the caching
                    # allocator reuses blocks; empty_cache only adds sync + churn.

                    if nxt_prepared is None:
                        break
                    prepared, upload_done = nxt_prepared

            ok = True
        finally:
            # Unblock the decoder thread: on an error path it may sit in
            # frame_q.put() on a full queue, and draining lets stream_frames'
            # generator finally reap its ffmpeg child instead of leaving the
            # thread and process alive until interpreter exit.
            while True:
                try:
                    frame_q.get_nowait()
                except _queue.Empty:
                    break
            if not ok:
                # Encoder cleanup on a failed run: without this the .tmp.mp4
                # survived (looking like a valid output) and the encoder
                # subprocess outlived the exception.
                writer.abort()
                torch.cuda.empty_cache()

        if prof:
            prof.flush()

        # Bound the wait but do not ignore a straggler: the decoder thread is a
        # daemon, so if it is still blocked in proc.stdout.read (ffmpeg slowly
        # draining a pipe the encoder no longer consumes) it would outlive the
        # conversion holding an ffmpeg process. FrameReader.close() is the
        # cooperative hook; the generator's own finally is the real guarantee.
        decoder.join(timeout=5)
        if decoder.is_alive():
            logger.warning("解码线程未在 5s 内退出（帧队列已排空）；"
                           "该线程为 daemon，进程退出时会一并回收")
        reader.close()
        torch.cuda.empty_cache()

        # Close and mux audio
        source = input_path if reader.has_audio else None
        if want_hdr:
            writer.close(audio_source=source)
        else:
            writer.close(source_video=source)

        total_elapsed = time.time() - total_start
        if not frame_times:
            # Should be unreachable (probe_video rejects 0-frame inputs), but
            # np.mean([]) is nan and 1.0/nan must never reach the UI.
            raise RuntimeError(
                f"没有处理任何帧（n_frames={n_frames}）: {input_path}")
        avg_frame_time = max(float(np.mean(frame_times)), 1e-6)

        return {
            "total_elapsed": total_elapsed,
            "avg_frame_time": avg_frame_time,
            "fps": 1.0 / avg_frame_time,
            "n_frames": len(frame_times),
            "output_path": str(output_path),
            "output_size": (out_w, out_h),
            "hdr": want_hdr,
        }
