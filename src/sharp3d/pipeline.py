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

        # Load model
        self.predictor = SharpPredictor(
            device=device,
            use_compile=use_compile,
            use_fp16=use_fp16,
        )
        self._warmed_up = False

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
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """Process a single image → SBS output.

        Args:
            input_path: Path to input image.
            output_path: Path for SBS output image.
            output_depth: Also save depth map visualization.
            progress_callback: Optional callback for status updates.

        Returns:
            dict with timing info and output paths.
        """
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
        sbs_np = sbs_img.cpu().numpy()
        Image.fromarray(sbs_np).save(output_path)

        result = {
            "elapsed": elapsed,
            "fps": 1.0 / elapsed,
            "output_path": str(output_path),
            "output_size": (sw * 2, sh),
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
        crf: int = 18,
        hdr_output: bool | None = None,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> dict:
        """Process video → SBS video with audio.

        Args:
            input_path: Path to input video.
            output_path: Path for SBS output video.
            codec: "h264", "h265", or "av1".
            crf: Quality (lower = better, 18 = visually lossless).
            hdr_output: True = force HDR10 output, False = force SDR,
                        None = auto (HDR10 if the input is HDR).
            progress_callback: (frame_idx, total_frames, fps) callback.

        Returns:
            dict with timing info and output path.
        """
        from .hdr import FrameReader, Hdr10Writer, probe_video

        input_path = Path(input_path)
        output_path = Path(output_path)

        info = probe_video(input_path)
        reader = FrameReader(input_path, info)  # tone-maps HDR input to SDR
        n_frames = reader.n_frames
        vid_fps = reader.fps
        f_px = reader.width * 1.2  # ~60° FOV estimate

        # HDR output: explicit flag, or auto-match the input's HDR status
        is_hdr = info["is_hdr"]
        want_hdr = is_hdr if hdr_output is None else hdr_output
        if want_hdr and codec == "h264":
            codec = "h265"  # H.264 cannot carry HDR10

        if want_hdr:
            writer = Hdr10Writer(
                output_path, fps=vid_fps, width=reader.width * 2,
                height=reader.height, codec=codec, crf=crf,
            )
        else:
            writer = VideoWriter(
                output_path, fps=vid_fps,
                width=reader.width * 2, height=reader.height,
                codec=codec, crf=crf,
            )

        frame_times = []
        total_start = time.time()

        for i, frame in enumerate(reader.stream_frames()):
            img_resized, df, intrinsics_resized, (orig_w, orig_h) = prepare_input(
                frame, f_px, self.device
            )

            self._ensure_warmup(img_resized, df)

            torch.cuda.synchronize()
            t0 = time.time()

            g_ndc = self.predictor.predict(img_resized, df)
            g_world = fast_unproject(
                g_ndc,
                torch.eye(4, device=self.device),
                intrinsics_resized,
                INTERNAL_SHAPE,
                decompose_method=self.decompose_method,
            )
            sbs_img, _ = render_sbs(
                g_world, f_px, orig_w, orig_h, ipd=self.ipd
            )

            torch.cuda.synchronize()
            dt = time.time() - t0
            frame_times.append(dt)

            sbs_np = sbs_img.cpu().numpy()
            if want_hdr:
                writer.write_frame(sbs_np)
            else:
                writer.append_frame(sbs_np)

            if progress_callback:
                progress_callback(i, n_frames, 1.0 / dt)

            del g_world, g_ndc, sbs_img, img_resized, frame
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
            "output_size": (reader.width * 2, reader.height),
            "hdr": want_hdr,
        }
