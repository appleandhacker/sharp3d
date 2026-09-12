"""Export the FULL RGBGaussianPredictor (ViTs + SPN + decoder + composer) to ONNX.

One static graph replaces the current three-way split (ORT patch encoder +
ORT image encoder + torch.compile SPN/decoder), so TensorRT can fuse kernels
across the SPN boundary and INT8 QDQ can be applied to the ViT MatMuls.

The graph is traceable by construction (sharp's split/merge are atomic, the
ViT captures intermediates without hooks, DepthAlignment bakes the depth=None
branch). Exported in FP32 — quantize_static needs an FP32 base graph; TRT
applies FP16/INT8 precision afterwards.

Usage:
    python tests/export_full_model.py --perf quality   # 35 patches
    python tests/export_full_model.py --perf speed     # 21 patches
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from sharp.models import PredictorParams, create_predictor
from sharp3d import resolve_cache_dir


class _ExportWrapper(torch.nn.Module):
    """Flatten the Gaussians3D NamedTuple into 5 ordered tensor outputs."""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image, disparity_factor):
        g = self.predictor(image, disparity_factor)
        return (g.mean_vectors, g.singular_values, g.quaternions,
                g.colors, g.opacities)


def build_predictor(device, perf: str):
    params = PredictorParams()
    if perf == "speed":
        params.monodepth.use_patch_overlap = False
    predictor = create_predictor(params)
    ckpt = resolve_cache_dir() / "sharp_fp16.pt"
    if ckpt.exists():
        sd = torch.load(str(ckpt), map_location="cpu", mmap=True, weights_only=True)
    else:
        sd = torch.hub.load_state_dict_from_url(
            "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
            progress=False, map_location="cpu")
    predictor.load_state_dict(sd)   # fp16 values are cast up to fp32 params
    del sd
    return predictor.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", choices=["quality", "speed"], default="quality")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    n_patches = 35 if args.perf == "quality" else 21
    device = torch.device("cuda")
    t0 = time.time()
    print(f"[1/3] 构建模型 (perf={args.perf}, {n_patches} patches)…", flush=True)
    predictor = build_predictor(device, args.perf)
    wrapper = _ExportWrapper(predictor)
    print(f"      完成 ({time.time()-t0:.0f}s)", flush=True)

    onnx_path = resolve_cache_dir() / "onnx" / f"full_model_{n_patches}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    image = torch.randn(1, 3, 1536, 1536, device=device)
    df = torch.tensor([1.0], device=device, dtype=torch.float32)

    print(f"[2/3] 导出 ONNX (opset 17, 静态形状) -> {onnx_path}", flush=True)
    t1 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (image, df),
            str(onnx_path),
            opset_version=17,
            input_names=["image", "disparity_factor"],
            output_names=["mean_vectors", "singular_values", "quaternions",
                          "colors", "opacities"],
            dynamo=False,       # legacy tracer: split/merge loops need it
            do_constant_folding=True,
        )
    print(f"      导出完成 ({time.time()-t1:.0f}s, "
          f"{onnx_path.stat().st_size/2**20:.0f}MB)", flush=True)

    if args.skip_verify:
        return

    print("[3/3] 数值验证 (ONNX CUDA EP vs PyTorch fp32, 真实帧)…", flush=True)
    import shutil
    import subprocess
    import tempfile

    import onnxruntime as ort

    ffmpeg = (shutil.which("ffmpeg")
              or r"C:/Program Files/ffmpeg/bin/ffmpeg.exe")
    video = ROOT.parent / "outputs" / "_bench_4k.mp4"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_png = Path(tmp.name)
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video),
                    "-vf", "select=eq(n\\,10)", "-frames:v", "1",
                    str(tmp_png)], check=True)
    from PIL import Image
    frame = np.asarray(Image.open(tmp_png).convert("RGB"))
    tmp_png.unlink(missing_ok=True)

    from sharp3d.unproject import prepare_input
    img_r, df_t, _ir, _sz = prepare_input(frame, 1920 * 1.2, device)
    torch.cuda.synchronize()
    img_np = img_r.cpu().numpy().astype(np.float32)
    df_np = df_t.cpu().numpy().astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CUDAExecutionProvider"])
    ort_outs = sess.run(None, {"image": img_np, "disparity_factor": df_np})

    with torch.no_grad():
        ref = wrapper(torch.from_numpy(img_np).to(device),
                      torch.from_numpy(df_np).to(device))
    names = ["mean_vectors", "singular_values", "quaternions", "colors", "opacities"]
    ok = True
    for name, o, r in zip(names, ort_outs, ref):
        r_np = r.float().cpu().numpy()
        diff = float(np.abs(o - r_np).max())
        scale = float(np.abs(r_np).max())
        status = "OK" if diff < max(1e-2, scale * 1e-3) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {name:16s} shape={r_np.shape} max_abs_diff={diff:.3e} "
              f"(scale {scale:.3f})  {status}", flush=True)
    print("VERIFY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
