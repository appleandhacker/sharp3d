"""FP8 QDQ graph surgery on the patch encoder ONNX (opset 17 → 21).

Transforms the 96 weight GEMMs (qkv/proj/fc1/fc2) into explicit FP8 QDQ:
    Q(x, sx) → DQ → MatMul ← DQ(W_fp8, sw)
TRT 10+ on sm120 compiles these into real FP8 tensor-core GEMMs (probe B
confirmed TRT EP accepts opset-21 float8 QDQ). Dynamic attention MatMuls
(q@kT, attn@v) are left in FP16.

Activation scales are calibrated on real frames: a debug copy of the graph
exposes each target MatMul's input as an output, amax is collected over N
runs, sx = amax/448. Weight scales: sw = amax(|W|)/448, per-tensor (TRT
requires scalar scales for FP8).

Output: .cache/onnx/patch_encoder_fp8.onnx (single file, ~0.4GB)
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FP8_MAX = 448.0


def real_patches(device, n_runs: int):
    """Real normalized patches via the production preprocessing path."""
    import torch
    from sharp.models import normalizers
    from sharp.models.encoders.spn_encoder import split
    from sharp3d.unproject import prepare_input
    from verify_full_onnx import grab_frame

    norm = normalizers.AffineRangeNormalizer(
        input_range=(0, 1), output_range=(-1, 1)).to(device).eval()
    video = ROOT.parent / "outputs" / "_bench_4k.mp4"
    for i in range(n_runs):
        frame = grab_frame(video, i * 3)
        img_r, _df, _ir, _sz = prepare_input(frame, 1920 * 1.2, device)
        with torch.no_grad():
            x = norm(img_r)
            x1 = torch.nn.functional.interpolate(x, scale_factor=0.5,
                                                 mode="bilinear",
                                                 align_corners=False)
            x2 = torch.nn.functional.interpolate(x, scale_factor=0.25,
                                                 mode="bilinear",
                                                 align_corners=False)
            yield torch.cat((split(x, 0.25, 384), split(x1, 0.5, 384),
                             x2), dim=0).cpu().numpy()


def main():
    device = "cuda"
    onnx_path = ROOT / ".cache" / "onnx" / "patch_encoder.onnx"
    out_path = ROOT / ".cache" / "onnx" / "patch_encoder_fp8.onnx"

    print("[1/5] 加载图 …", flush=True)
    model = onnx.load(str(onnx_path))
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    # 目标 MatMul: input[1] 是初始器（权重 GEMM）
    targets = [n for n in g.node if n.op_type == "MatMul"
               and n.input[1] in inits]
    print(f"  目标 FP8 GEMM: {len(targets)} / "
          f"{sum(1 for n in g.node if n.op_type == 'MatMul')}", flush=True)

    print("[2/5] opset 17 → 21 …", flush=True)
    from onnx import version_converter
    model = version_converter.convert_version(model, 21)
    g = model.graph

    print("[3/5] 校准激活 amax（调试图暴露 96 个 GEMM 输入）…", flush=True)
    # 调试图: 把每个目标 MatMul 的 input[0] 加为输出
    dbg_outputs = [helper.make_tensor_value_info(
        t.input[0], TensorProto.FLOAT, None) for t in targets]
    dbg_graph = helper.make_graph(
        list(g.node), g.name + "_dbg",
        [vi for vi in g.input if vi.name not in
         {t.name for t in g.initializer}],
        dbg_outputs, initializer=g.initializer)
    dbg_model = helper.make_model(
        dbg_graph, opset_imports=[helper.make_opsetid("", 21)])
    dbg_model.ir_version = model.ir_version
    dbg_path = ROOT / ".cache" / "onnx" / "_patch_dbg.onnx"
    onnx.save(dbg_model, str(dbg_path))

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(dbg_path), sess_options=so,
                                providers=["CUDAExecutionProvider"])
    amax = np.zeros(len(targets), dtype=np.float64)
    n_runs = 6
    for i, patches in enumerate(real_patches(device, n_runs)):
        outs = sess.run(None, {"patches": patches.astype(np.float32)})
        for j, o in enumerate(outs):
            amax[j] = max(amax[j], float(np.abs(o).max()))
        print(f"  run {i+1}/{n_runs}", flush=True)
    sx_map = {t.input[0]: np.float32(max(a / FP8_MAX, 1e-12))
              for t, a in zip(targets, amax)}
    dbg_path.unlink(missing_ok=True)

    print("[4/5] 插入 FP8 Q/DQ 节点 …", flush=True)
    import torch
    new_nodes = []
    new_inits = []
    for t in targets:
        w_init = inits[t.input[1]]
        w = numpy_helper.to_array(w_init).astype(np.float32)
        sw = np.float32(max(np.abs(w).max() / FP8_MAX, 1e-12))
        # numpy 无 float8 dtype：经 torch 转换，raw bytes 与 FLOAT8E4M3FN 一致
        w8_u8 = torch.from_numpy(w).div(sw).clamp(-FP8_MAX, FP8_MAX) \
            .to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        w8 = numpy_helper.from_array(w8_u8, t.input[1] + "_fp8")
        w8.data_type = TensorProto.FLOAT8E4M3FN

        q_in = t.name + "_xq"
        dq_in = t.name + "_xdq"
        dq_w = t.input[1] + "_qd"
        w8_name = t.input[1] + "_fp8"
        sx, sw_t = sx_map[t.input[0]], np.float32(sw)

        new_nodes.extend([
            helper.make_node("QuantizeLinear", [t.input[0], f"sx_{t.name}"],
                             [q_in],
                             output_dtype=TensorProto.FLOAT8E4M3FN,
                             name=t.name + "_Qx"),
            helper.make_node("DequantizeLinear", [q_in, f"sx_{t.name}"],
                             [dq_in], name=t.name + "_DQx"),
            helper.make_node("DequantizeLinear", [w8_name, f"sw_{t.name}"],
                             [dq_w], name=t.name + "_DQw"),
        ])
        new_inits.extend([
            w8,
            numpy_helper.from_array(np.array(sx, np.float32), f"sx_{t.name}"),
            numpy_helper.from_array(np.array(sw, np.float32), f"sw_{t.name}"),
        ])
        # 重接 MatMul 输入
        t.input[0] = dq_in
        t.input[1] = dq_w

    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)

    print("[5/5] 保存 …", flush=True)
    onnx.checker.check_model(model)   # 单文件应 <2GB
    onnx.save(model, str(out_path))
    print(f"完成: {out_path} ({out_path.stat().st_size/2**20:.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
