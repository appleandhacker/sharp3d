"""INT8 QDQ static quantization of the two ViT encoders (standalone graphs).

The full-graph calibration OOMs a 12GB GPU (QDQ transient copies of the
35-batch ViT activations co-resident with the rest of the graph). The ViT
activation ranges depend only on the ViT inputs, so calibrating the
standalone ONNX files gives identical scales at a fraction of the memory.

Calibration inputs are REAL patches: pyramid+split of decoded video frames,
exactly what production feeds the encoders.

Usage:
    python tests/quantize_vit_int8.py --frames 24
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sharp3d import resolve_cache_dir  # noqa: E402


def _ffmpeg_frames(video: Path, n: int):
    import shutil
    import subprocess
    import tempfile
    from PIL import Image

    ffmpeg = shutil.which("ffmpeg") or r"C:/Program Files/ffmpeg/bin/ffmpeg.exe"
    # 源视频只有 ~24 帧，逐帧取样（i*7 会越界产出空 PNG）
    for i in range(n):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_png = Path(tmp.name)
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video),
                            "-vf", f"select=eq(n\\,{i})", "-frames:v", "1",
                            str(tmp_png)], capture_output=True)
        if not tmp_png.exists() or tmp_png.stat().st_size == 0:
            tmp_png.unlink(missing_ok=True)
            continue   # 越界帧：跳过
        frame = np.asarray(Image.open(tmp_png).convert("RGB"))
        tmp_png.unlink(missing_ok=True)
        yield frame


def _make_reader(video: Path, n_frames: int, which: str, chunk: int = 7):
    """Yield calibration dicts through the production preprocessing path.

    The ViT ONNX graphs have dynamic batch, and per-tensor activation ranges
    are batch-independent — feeding (35,3,384,384) in chunks of 7 avoids the
    QDQ transient OOM while covering the same tensors.
    """
    import torch
    from sharp.models.encoders.spn_encoder import split
    from sharp3d.unproject import prepare_input

    dev = torch.device("cuda")
    # 生产路径中 ViT 输入是归一化后的图像（(0,1)→(-1,1)，见
    # MonodepthWithEncodingAdaptor.forward），校准必须走同一分布。
    from sharp.models import normalizers
    norm = normalizers.AffineRangeNormalizer(input_range=(0, 1),
                                             output_range=(-1, 1)).to(dev).eval()
    for frame in _ffmpeg_frames(video, n_frames):
        img_r, _df, _ir, _sz = prepare_input(frame, 1920 * 1.2, dev)
        torch.cuda.synchronize()
        with torch.no_grad():
            x = norm(img_r)
            x1 = torch.nn.functional.interpolate(x, scale_factor=0.5,
                                                 mode="bilinear",
                                                 align_corners=False)
            x2 = torch.nn.functional.interpolate(x, scale_factor=0.25,
                                                 mode="bilinear",
                                                 align_corners=False)
            if which == "patch":
                x = torch.cat((split(x, overlap_ratio=0.25, patch_size=384),
                               split(x1, overlap_ratio=0.5, patch_size=384),
                               x2), dim=0)          # (35,3,384,384)
            else:
                x = x2                              # (1,3,384,384)
        x = x.cpu()
        for i in range(0, x.shape[0], chunk):
            yield {"patches": x[i:i + chunk].numpy().astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--video", default=None)
    ap.add_argument("--method", choices=["MinMax", "Percentile", "Entropy"],
                    default="MinMax")
    ap.add_argument("--percentile", type=float, default=99.999)
    args = ap.parse_args()

    onnx_dir = resolve_cache_dir() / "onnx"
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

    method = getattr(CalibrationMethod, args.method)
    extra = {
        "ActivationSymmetric": True,
        "CalibTensorRangeSymmetric": True,
    }
    if args.method == "Percentile":
        extra["CalibPercentile"] = args.percentile
    for which in ("patch", "image"):
        base = onnx_dir / f"{which}_encoder.onnx"
        qdq = onnx_dir / f"{which}_encoder_int8.onnx"
        if not base.exists():
            print(f"跳过 {which}: {base} 不存在")
            continue
        print(f"[{which}] 校准 + QDQ ({args.frames} 帧, {args.method})…",
              flush=True)
        t0 = time.time()
        quantize_static(
            model_input=str(base),
            model_output=str(qdq),
            calibration_data_reader=_Reader(
                _make_reader(video, args.frames, which)),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            calibration_providers=["CUDAExecutionProvider"],
            extra_options=extra,
        )
        print(f"[{which}] 完成 ({time.time()-t0:.0f}s) -> "
              f"{qdq.stat().st_size / 2**20:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
