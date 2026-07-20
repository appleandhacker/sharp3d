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
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> dict:
        """Process video → SBS video with audio.

        Args:
            input_path: Path to input video.
            output_path: Path for SBS output video.
            codec: "h264", "h265", or "av1".
            crf: Quality (lower = better, 18 = visually lossless).
            progress_callback: (frame_idx, total_frames, fps) callback.

        Returns:
            dict with timing info and output path.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        reader = VideoReader(input_path)
        n_frames = reader.n_frames
        vid_fps = reader.fps

        # Get focal length from first frame
        first_frame = reader.get_frame(0)
        _, _, f_px = sharp_io.load_rgb(input_path)  # Get f_px from metadata
        # For video, estimate f_px from aspect ratio if not available
        if f_px is None:
            f_px = reader.width * 1.2  # Default ~60° FOV

        writer = VideoWriter(
            output_path, fps=vid_fps,
            width=reader.width * 2, height=reader.height,
            codec=codec, crf=crf,
        )

        frame_times = []
        total_start = time.time()

        for i in range(n_frames):
            frame = reader.get_frame(i)
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

            writer.append_frame(sbs_img.cpu().numpy())

            if progress_callback:
                progress_callback(i, n_frames, 1.0 / dt)

            del g_world, g_ndc, sbs_img, img_resized, frame
            torch.cuda.empty_cache()

        # Close and mux audio
        source = input_path if reader.has_audio else None
        writer.close(source_video=source)
        reader.close()

        total_elapsed = time.time() - total_start
        avg_frame_time = np.mean(frame_times)

        return {
            "total_elapsed": total_elapsed,
            "avg_frame_time": avg_frame_time,
            "fps": 1.0 / avg_frame_time,
            "n_frames": n_frames,
            "output_path": str(output_path),
            "output_size": (reader.width * 2, reader.height),
        }
