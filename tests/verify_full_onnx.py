"""Verify the exported full-model ONNX against PyTorch on a REAL frame.

Stage-wise debug showed the divergence is cumulative fp noise amplified by
the composer's z-division (outlier pixels), so the acceptance criterion is
per-field max-abs-diff thresholds calibrated on real data, plus a rendered-
image PSNR check is done separately in bench_quality.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from export_full_model import build_predictor, _ExportWrapper  # noqa: E402
from sharp3d import resolve_cache_dir  # noqa: E402


def grab_frame(video: Path, idx: int) -> np.ndarray:
    import shutil
    ffmpeg = shutil.which("ffmpeg") or r"C:/Program Files/ffmpeg/bin/ffmpeg.exe"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_png = Path(tmp.name)
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video),
                    "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                    str(tmp_png)], check=True)
    from PIL import Image
    frame = np.asarray(Image.open(tmp_png).convert("RGB"))
    tmp_png.unlink(missing_ok=True)
    return frame


def main():
    device = torch.device("cuda")
    perf = sys.argv[1] if len(sys.argv) > 1 else "quality"
    n_patches = 35 if perf == "quality" else 21

    predictor = build_predictor(device, perf)
    wrapper = _ExportWrapper(predictor)

    onnx_path = resolve_cache_dir() / "onnx" / f"full_model_{n_patches}.onnx"
    video = ROOT.parent / "outputs" / "_bench_4k.mp4"

    from sharp3d.unproject import prepare_input
    frame = grab_frame(video, 10)
    img_r, df_t, _ir, _sz = prepare_input(frame, 1920 * 1.2, device)
    torch.cuda.synchronize()
    img_np = img_r.cpu().numpy().astype(np.float32)
    df_np = df_t.cpu().numpy().astype(np.float32)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CUDAExecutionProvider"])
    ort_outs = sess.run(None, {"image": img_np, "disparity_factor": df_np})
    with torch.no_grad():
        ref = wrapper(torch.from_numpy(img_np).to(device),
                      torch.from_numpy(df_np).to(device))

    # 阈值依据: mean 吻合度是主要指标; max 允许离群像素 (z 除法放大)
    limits = {
        "mean_vectors":    (2e-4, 0.05),   # (mean_abs, max_abs)
        "singular_values": (1e-5, 0.01),
        "quaternions":     (1e-3, 1.0),
        "colors":          (1e-3, 0.30),
        "opacities":       (1e-3, 0.30),
    }
    names = ["mean_vectors", "singular_values", "quaternions",
             "colors", "opacities"]
    ok_all = True
    for name, o, r in zip(names, ort_outs, ref):
        r_np = r.float().cpu().numpy()
        ad = np.abs(o - r_np)
        m_lim, x_lim = limits[name]
        good = ad.mean() < m_lim and ad.max() < x_lim
        ok_all &= good
        print(f"  {name:16s} mean={ad.mean():.3e} (<{m_lim:.0e}) "
              f"max={ad.max():.3e} (<{x_lim}) ref_absmax={np.abs(r_np).max():.3f} "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print("VERIFY:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
