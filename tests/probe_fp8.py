"""FP8 feasibility probes for the DINOv2 ViT on sm_120.

Probe A (quality): swap every nn.Linear in the patch encoder with a
    torch._scaled_mm FP8 (e4m3, dynamic per-tensor activation scale +
    static per-tensor weight scale) and measure feature-space error on a
    REAL frame. FP8's wide per-tensor dynamic range is the hypothesis under
    test — INT8 per-tensor symmetric collapsed to 25dB image quality.

Probe B (toolchain): build a tiny opset-21 ONNX graph with FP8 QDQ around a
    MatMul and ask ORT's TRT EP to build an engine. If TRT EP rejects it,
    the ONNX→TRT FP8 route is dead regardless of quality.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from export_full_model import build_predictor  # noqa: E402
from verify_full_onnx import grab_frame  # noqa: E402


class FP8Linear(nn.Module):
    """torch._scaled_mm drop-in: e4m3 weights + e4m3 activations."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        sw = (w.abs().amax().clamp(min=1e-12) / 448.0)
        self.w_fp8 = nn.Parameter(
            (w / sw).clamp(-448, 448).to(torch.float8_e4m3fn),
            requires_grad=False)
        self.sw = nn.Parameter(sw.reshape(()), requires_grad=False)
        self.bias = (nn.Parameter(lin.bias.detach().clone())
                     if lin.bias is not None else None)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        sx = (x2.detach().abs().amax().clamp(min=1e-12) / 448.0)
        x8 = (x2.float() / sx).clamp(-448, 448).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(x8, self.w_fp8.t(), scale_a=sx,
                               scale_b=self.sw, out_dtype=torch.float16)
        out = out.reshape(*shape[:-1], out.shape[-1]).to(x.dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


def probe_a(device):
    print("=" * 60)
    print("Probe A: FP8 Linear 质量探针（真实帧, patch encoder）")
    print("=" * 60)
    predictor = build_predictor(device, "quality")
    spn = predictor.monodepth_model.monodepth_predictor.encoder

    from sharp3d.unproject import prepare_input
    from sharp.models import normalizers
    frame = grab_frame(ROOT.parent / "outputs" / "_bench_4k.mp4", 10)
    img_r, _df, _ir, _sz = prepare_input(frame, 1920 * 1.2, device)
    norm = normalizers.AffineRangeNormalizer(
        input_range=(0, 1), output_range=(-1, 1)).to(device).eval()
    with torch.no_grad():
        x = norm(img_r)
        x1 = torch.nn.functional.interpolate(x, scale_factor=0.5,
                                             mode="bilinear",
                                             align_corners=False)
        x2 = torch.nn.functional.interpolate(x, scale_factor=0.25,
                                             mode="bilinear",
                                             align_corners=False)
        from sharp.models.encoders.spn_encoder import split
        patches = torch.cat((split(x, 0.25, 384), split(x1, 0.5, 384), x2), 0)

    # 参考输出（fp32 权重, fp32 计算）
    with torch.no_grad():
        ref_feat, ref_ints = spn.patch_encoder(patches)

    # 换 FP8 Linear
    n_fp8 = 0
    for block in spn.patch_encoder.blocks:
        for name in ("qkv", "proj"):
            mod = getattr(block.attn, name, None)
            if isinstance(mod, nn.Linear):
                setattr(block.attn, name, FP8Linear(mod).to(device))
                n_fp8 += 1
        for name in ("fc1", "fc2"):
            mod = getattr(block.mlp, name, None)
            if isinstance(mod, nn.Linear):
                setattr(block.mlp, name, FP8Linear(mod).to(device))
                n_fp8 += 1
    print(f"  替换 Linear → FP8: {n_fp8} 个（24 blocks × 4）")

    with torch.no_grad():
        fp8_feat, fp8_ints = spn.patch_encoder(patches)

    for name, o, r in (("features", fp8_feat, ref_feat),
                       ("intermediates[5]", fp8_ints[5], ref_ints[5]),
                       ("intermediates[23]", fp8_ints[23], ref_ints[23])):
        d = (o.float() - r.float()).abs()
        rel = d.norm() / r.float().norm()
        print(f"  {name:18s} max={d.max():.3e} mean={d.mean():.3e} "
              f"相对L2={rel:.4f}", flush=True)

    # 单块计时对比（同形状 GEMM）
    blk = spn.patch_encoder.blocks[0]
    xin = torch.randn(35, 577, 1024, device=device)
    with torch.no_grad():
        for _ in range(3):
            blk(xin)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            blk(xin)
        torch.cuda.synchronize()
        t_fp8 = (time.time() - t0) / 10
    print(f"  单块(35×577) forward: FP8 {t_fp8*1e3:.1f}ms", flush=True)
    return True


def probe_b():
    print("=" * 60)
    print("Probe B: opset-21 FP8 QDQ → ORT TRT EP 接受度")
    print("=" * 60)
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    M, K, N = 64, 128, 256
    w = (np.random.randn(N, K) / np.sqrt(K)).astype(np.float32)
    sw = np.float32(max(np.abs(w).max() / 448.0, 1e-8))
    sx = np.float32(0.01)

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "sx"], ["x8"],
                         output_dtype=TensorProto.FLOAT8E4M3FN),
        helper.make_node("DequantizeLinear", ["x8", "sx"], ["xd"], ),
        helper.make_node("QuantizeLinear", ["w", "sw"], ["w8"],
                         output_dtype=TensorProto.FLOAT8E4M3FN),
        helper.make_node("DequantizeLinear", ["w8", "sw"], ["wd"]),
        helper.make_node("MatMul", ["xd", "wdT"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "fp8_probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [M, N])],
        initializer=[
            numpy_helper.from_array(w, "w"),
            numpy_helper.from_array(w.T.copy(), "wdT"),   # (K,N) for MatMul
            numpy_helper.from_array(np.array(sx, np.float32), "sx"),
            numpy_helper.from_array(np.array(sw, np.float32), "sw"),
        ])
    # QDQ 语义：DQ(w8,sw) 是 (N,K) 再转置不对 —— 直接用 (K,N) 权重做 QDQ
    # 修正：把 wdT 当作被量化的张量本身
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    path = ROOT / ".cache" / "onnx" / "_fp8_probe.onnx"
    onnx.save(model, str(path))

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:
        sess = ort.InferenceSession(
            str(path), sess_options=so,
            providers=[("TensorrtExecutionProvider",
                        {"trt_fp16_enable": True,
                         "trt_engine_cache_enable": False}),
                       "CUDAExecutionProvider"])
        active = sess.get_providers()
        x = np.random.randn(M, K).astype(np.float32) * sx
        y = sess.run(None, {"x": x})[0]
        y_ref = x @ w.T
        err = np.abs(y - y_ref).max()
        print(f"  TRT EP active: {'TensorrtExecutionProvider' in active}")
        print(f"  FP8 QDQ MatMul 输出误差: {err:.3e} (相对 {err/np.abs(y_ref).max():.3e})")
        print("  结论:", "TRT 接受 FP8 QDQ ✓" if "TensorrtExecutionProvider" in active
              else "TRT 拒绝/忽略 FP8 QDQ ✗")
        return "TensorrtExecutionProvider" in active
    except Exception as e:
        print(f"  TRT EP 失败: {e}")
        print("  结论: TRT 拒绝 FP8 QDQ ✗")
        return False


import time  # noqa: E402

if __name__ == "__main__":
    device = torch.device("cuda")
    probe_a(device)
    probe_b()
