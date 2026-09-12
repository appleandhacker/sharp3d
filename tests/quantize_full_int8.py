"""INT8 static quantization (explicit QDQ) of the full SHARP ONNX model.

ORT TRT EP only builds INT8 kernels when the graph carries explicit
QuantizeLinear/DequantizeLinear nodes ("explicit quantization"). The implicit
path (trt_int8_enable + calibration table) never engaged for this ViT because
the ONNX graph has no QDQ nodes — measured: INT8 fell back to FP16 everywhere.

Strategy:
    - Quantize ONLY MatMul nodes (the ViT qkv/proj/mlp GEMMs, measured 2.8x
      INT8 tensor-core throughput on this GPU). Convolutions (SPN, decoder,
      gaussian head) stay FP16 — depth regression is quantization-sensitive
      and the convs are not the bottleneck.
    - Per-channel weight scales, per-tensor symmetric activation scales
      (TRT-friendly), MinMax calibration over ~32 real frames.
    - Calibrate on CUDA EP (CPU would take minutes per frame).

Usage:
    python tests/quantize_full_int8.py --perf quality [--frames 32]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sharp3d import resolve_cache_dir  # noqa: E402


def make_calibration_reader(video: Path, n_frames: int, f_px: float = 1920 * 1.2):
    """Yield real-frame inputs through the production preprocessing path."""
    import shutil
    import subprocess
    import tempfile

    import torch
    from PIL import Image

    from sharp3d.unproject import prepare_input

    ffmpeg = (shutil.which("ffmpeg")
              or r"C:/Program Files/ffmpeg/bin/ffmpeg.exe")
    dev = torch.device("cuda")
    for i in range(n_frames):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_png = Path(tmp.name)
        sel = f"select=eq(n\\,{i * 7})"
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video),
                        "-vf", sel, "-frames:v", "1", str(tmp_png)],
                       check=True)
        frame = np.asarray(Image.open(tmp_png).convert("RGB"))
        tmp_png.unlink(missing_ok=True)
        img_r, df, _ir, _sz = prepare_input(frame, f_px, dev)
        torch.cuda.synchronize()
        yield {
            "image": img_r.cpu().numpy().astype(np.float32),
            "disparity_factor": df.cpu().numpy().astype(np.float32),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", choices=["quality", "speed"], default="quality")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--video", default=None,
                    help="calibration video (default: ../outputs/_bench_4k.mp4)")
    args = ap.parse_args()

    n_patches = 35 if args.perf == "quality" else 21
    onnx_dir = resolve_cache_dir() / "onnx"
    base = onnx_dir / f"full_model_{n_patches}.onnx"
    if not base.exists():
        sys.exit(f"缺少 {base} — 先跑 tests/export_full_model.py --perf {args.perf}")
    qdq = onnx_dir / f"full_model_{n_patches}_int8.onnx"

    video = Path(args.video) if args.video else \
        ROOT.parent / "outputs" / "_bench_4k.mp4"
    if not video.exists():
        sys.exit(f"校准视频不存在: {video}")

    from onnxruntime.quantization import (CalibrationDataReader,
                                          CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    class _Reader(CalibrationDataReader):
        def __init__(self, gen):
            self._gen = gen

        def get_next(self):
            return next(self._gen, None)

    print(f"[1/2] 静态量化 ({args.frames} 帧真实数据校准, MatMul only, "
          f"per-channel 权重)…", flush=True)
    t0 = time.time()
    quantize_static(
        model_input=str(base),
        model_output=str(qdq),
        calibration_data_reader=_Reader(make_calibration_reader(video, args.frames)),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CUDAExecutionProvider"],
        use_external_data_format=True,   # QDQ 模型 >2GB，protobuf 单文件放不下
        extra_options={
            "ActivationSymmetric": True,   # TRT INT8 prefers symmetric
            "CalibTensorRangeSymmetric": True,
        },
    )
    print(f"      完成 ({time.time()-t0:.0f}s)\n"
          f"      QDQ 模型: {qdq} ({qdq.stat().st_size/2**20:.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
