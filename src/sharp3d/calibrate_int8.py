"""INT8 calibration for SHARP TensorRT acceleration.

Generates a TensorRT INT8 calibration table from representative video frames,
enabling INT8 quantized inference for the patch_encoder and image_encoder.

Usage:
    # Step 1: Generate calibration table from a video
    python -m sharp3d.calibrate_int8 --video input.mp4 --frames 50

    # Step 2: The calibration table is saved to .cache/trt/calibration_table
    # Step 3: Enable INT8 in the GUI speed mode (automatic if table exists)

Requirements:
    - tensorrt (installed as torch_tensorrt dependency)
    - onnxruntime-gpu with TensorrtExecutionProvider
    - A representative video (20-50 frames recommended)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
_ONNX_DIR = _CACHE_DIR / "onnx"
_TRT_CACHE_DIR = _CACHE_DIR / "trt"
_CALIBRATION_TABLE = _TRT_CACHE_DIR / "calibration_table"


class CalibrationDataReader:
    """Reads calibration frames from a video and yields preprocessed inputs."""

    def __init__(self, video_path: str, n_frames: int = 50,
                 input_shape: tuple = (35, 3, 384, 384)):
        self.video_path = video_path
        self.n_frames = n_frames
        self.input_shape = input_shape
        self._frames = []
        self._idx = 0
        self._load_frames()

    def _load_frames(self):
        """Extract and preprocess frames from the video."""
        from sharp3d.hdr import FrameReader
        from sharp3d.unproject import prepare_input, INTERNAL_SHAPE
        import torch

        reader = FrameReader(Path(self.video_path))
        total = reader.n_frames
        # Sample evenly across the video
        indices = np.linspace(0, total - 1, self.n_frames, dtype=int)

        device = torch.device("cuda")
        f_px = reader.width * 1.2

        logger.info("Loading %d calibration frames from %s (total: %d)",
                     self.n_frames, self.video_path, total)

        for idx in indices:
            frame = reader.read_frame(int(idx))
            img_r, df, ir, _ = prepare_input(frame, f_px, device)
            # img_r is [1, 3, 1536, 1536], we need patches for the ViT
            # For calibration, we just need representative ViT inputs
            # Use random patches from the image (simpler than running full SPN)
            self._frames.append(img_r.cpu().numpy())

        logger.info("Loaded %d calibration frames", len(self._frames))

    def get_next(self):
        """Return the next calibration input as a dict."""
        if self._idx >= len(self._frames):
            return None
        # For patch_encoder calibration, we need [batch, 3, 384, 384] inputs
        # Generate random patches from the full image
        img = self._frames[self._idx]
        self._idx += 1

        # Create patch-sized inputs by cropping from the full image
        # This gives representative activation distributions
        h, w = img.shape[2], img.shape[3]
        patches = []
        patch_size = 384
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = img[:, :, y:y+patch_size, x:x+patch_size]
                patches.append(patch)

        if not patches:
            # Image smaller than patch_size, just resize
            import torch
            import torch.nn.functional as F
            t = torch.from_numpy(img)
            t = F.interpolate(t, size=(patch_size, patch_size), mode='bilinear')
            patches = [t.numpy()]

        # Stack patches into a batch
        batch = np.concatenate(patches, axis=0).astype(np.float32)
        return {"patches": batch}

    def rewind(self):
        self._idx = 0


def calibrate_with_ort(onnx_path: Path, calibration_table_path: Path,
                       video_path: str, n_frames: int = 50):
    """Run ORT TensorRT INT8 calibration.

    Creates an ORT session with INT8 enabled and runs inference with
    calibration data. TensorRT collects activation statistics and
    generates the calibration table.
    """
    import onnxruntime as ort

    # Ensure cuDNN is in PATH
    torch_lib = Path(__import__("torch").__file__).parent / "lib"
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 2

    _TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Create session with INT8 calibration enabled
    providers = [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_int8_enable": True,
            "trt_int8_calibration_table_name": str(calibration_table_path),
            "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(_TRT_CACHE_DIR),
        }),
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]

    logger.info("Creating ORT session with INT8 calibration enabled...")
    logger.info("Calibration table: %s", calibration_table_path)

    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    active = session.get_providers()
    logger.info("Active providers: %s", active)

    if "TensorrtExecutionProvider" not in active:
        logger.error("TensorRT EP not available, cannot calibrate")
        return False

    # Create calibration data reader
    reader = CalibrationDataReader(video_path, n_frames)

    # Run inference with calibration data
    input_name = session.get_inputs()[0].name
    logger.info("Running calibration inference (%d frames)...", n_frames)

    for i in range(n_frames):
        data = reader.get_next()
        if data is None:
            break
        try:
            session.run(None, data)
            if (i + 1) % 10 == 0:
                logger.info("  Calibrated %d/%d frames", i + 1, n_frames)
        except Exception as e:
            logger.warning("Calibration frame %d failed: %s", i, e)

    logger.info("Calibration complete!")

    # Check if calibration table was generated
    if calibration_table_path.exists():
        size = calibration_table_path.stat().st_size
        logger.info("Calibration table saved: %s (%.1f KB)",
                     calibration_table_path, size / 1024)
        return True
    else:
        logger.warning("Calibration table not found at %s", calibration_table_path)
        logger.warning("TRT may have generated it in a different location")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate INT8 calibration table for SHARP TensorRT")
    parser.add_argument("--video", required=True,
                        help="Path to calibration video (20-50 frames recommended)")
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of frames to use for calibration (default: 50)")
    parser.add_argument("--encoder", choices=["patch", "image", "both"],
                        default="patch",
                        help="Which encoder to calibrate (default: patch)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    if args.encoder in ("patch", "both"):
        onnx_path = _ONNX_DIR / "patch_encoder.onnx"
        if not onnx_path.exists():
            logger.error("patch_encoder.onnx not found. Run the GUI first to export it.")
            sys.exit(1)

        table_path = _TRT_CACHE_DIR / "patch_encoder_calibration_table"
        logger.info("=== Calibrating patch_encoder ===")
        success = calibrate_with_ort(onnx_path, table_path, str(video_path), args.frames)
        if success:
            logger.info("patch_encoder INT8 calibration: SUCCESS")
        else:
            logger.error("patch_encoder INT8 calibration: FAILED")

    if args.encoder in ("image", "both"):
        onnx_path = _ONNX_DIR / "image_encoder.onnx"
        if not onnx_path.exists():
            logger.error("image_encoder.onnx not found. Run the GUI first to export it.")
            sys.exit(1)

        table_path = _TRT_CACHE_DIR / "image_encoder_calibration_table"
        logger.info("=== Calibrating image_encoder ===")
        success = calibrate_with_ort(onnx_path, table_path, str(video_path), args.frames)
        if success:
            logger.info("image_encoder INT8 calibration: SUCCESS")
        else:
            logger.error("image_encoder INT8 calibration: FAILED")


if __name__ == "__main__":
    main()
