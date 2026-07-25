"""Full pipeline: image/video → SHARP predict → unproject → SBS render → output.

Orchestrates all modules into a single coherent pipeline.
"""

import gc
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
        self.predictor = SharpPredictor(device=device)
        self._warmed_up = True  # SharpPredictor does warmup in __init__

    def _ensure_warmup(self, img_resized, disparity_factor):
        """Warmup on first call (triggers torch.compile)."""
        if not self._warmed_up:
            self.predictor.warmup(img_resized, disparity_factor)
            self._warmed_up = True

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

        self._ensure_warmup(img_resized, df)

        if progress_callback:
            progress_callback("Predicting 3D Gaussians...")

        torch.cuda.synchronize()
        t0 = time.time()

        g_ndc = self.predictor.predict(img_resized, df)

        if progress_callback:
            progress_callback("Unprojecting to world space...")

        g_world = fast_unproject(
            g_ndc,
            torch.eye(4, device=self.device),
            intrinsics_resized,
            INTERNAL_SHAPE,
            decompose_method=self.decompose_method,
        )

        if progress_callback:
            progress_callback("Rendering SBS...")

        sbs_img, (sw, sh) = render_sbs(
            g_world, f_px, orig_w, orig_h, ipd=self.ipd
        )

        torch.cuda.synchronize()
        elapsed = time.time() - t0

        # Save output
        packed = pack_stereo(format, sbs_img)
        sbs_np = packed.cpu().numpy()
        Image.fromarray(sbs_np).save(output_path)

        result = {
            "elapsed": elapsed,
            "fps": 1.0 / elapsed,
            "output_path": str(output_path),
            "output_size": output_size(format, sw, sh),
        }

        # Optional depth map
        if output_depth:
            depth_path = output_path.with_stem(output_path.stem + "_depth")
            depth_img = render_depth_map(g_world, f_px, orig_w, orig_h)
            Image.fromarray(depth_img.cpu().numpy()).save(depth_path)
            result["depth_path"] = str(depth_path)

        # Cleanup
        del g_world, g_ndc, sbs_img
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

        out_w, out_h = output_size(format, reader.width, reader.height)

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

        # Temporal stabilization (same as GUI path).
        from .temporal import TemporalStabilizer, KalmanScalar
        from .render import _compute_focus_depth_gpu
        stab = TemporalStabilizer(mode="adaptive", device=self.device)
        conv_kf = KalmanScalar(q_pos=0.05, q_vel=0.02, r=0.15)

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
            # CPU-side wait: the pinned host buffer must not be reused (by the
            # next frame's pin_memory) while the async copy still reads it.
            # Only the CPU stalls ~10ms; the GPU keeps rendering in parallel.
            upload_done.synchronize()
            return prepared, upload_done

        i = 0
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

                self._ensure_warmup(img_resized, df)

                t0 = time.time()

                g_ndc = self.predictor.predict(img_resized, df)
                stab.stabilize(g_ndc, img=img_resized)
                g_world = fast_unproject(
                    g_ndc,
                    torch.eye(4, device=self.device),
                    intrinsics_resized,
                    INTERNAL_SHAPE,
                    decompose_method=self.decompose_method,
                )
                # Convergence Kalman smoothing (auto mode).
                focus = _compute_focus_depth_gpu(g_world.mean_vectors)
                frame_conv = conv_kf.update(focus)
                sbs_img, _ = render_sbs(
                    g_world, f_px, orig_w, orig_h, ipd=self.ipd,
                    convergence=frame_conv,
                )

                torch.cuda.synchronize()
                dt = time.time() - t0
                frame_times.append(dt)

                packed = pack_stereo(format, sbs_img)
                sbs_np = packed.cpu().numpy()
                if want_hdr:
                    writer.write_frame(sbs_np)
                else:
                    writer.append_frame(sbs_np)

                if progress_callback:
                    progress_callback(i, n_frames, 1.0 / dt)
                i += 1

                del g_world, g_ndc, sbs_img, img_resized, prepared
                # No per-frame empty_cache(): constant shapes mean the caching
                # allocator reuses blocks; empty_cache only adds sync + churn.

                if nxt_prepared is None:
                    break
                prepared, upload_done = nxt_prepared

        while True:
            try:
                frame_q.get_nowait()
            except _queue.Empty:
                break
        decoder.join(timeout=5)
        torch.cuda.empty_cache()

        # Close and mux audio
        source = input_path if reader.has_audio else None
        if want_hdr:
            writer.close(audio_source=source)
        else:
            writer.close(source_video=source)

        total_elapsed = time.time() - total_start
        avg_frame_time = np.mean(frame_times)

        return {
            "total_elapsed": total_elapsed,
            "avg_frame_time": avg_frame_time,
            "fps": 1.0 / avg_frame_time,
            "n_frames": len(frame_times),
            "output_path": str(output_path),
            "output_size": (out_w, out_h),
            "hdr": want_hdr,
        }
