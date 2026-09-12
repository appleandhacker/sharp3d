"""Verify the _forward_numpy dtype fix without a full model.

Builds a tiny ONNX model with a float32 input, wraps it in an ORTEncoder-shaped
stub, and calls the real _forward_numpy with a float16 tensor. Before the fix
this raised a dtype mismatch (float16 ndarray into a float32-declared input);
after it must return a float32 tensor on the target device.
"""
import os
import sys
from pathlib import Path

SHARP3D_SRC = os.environ.get("SHARP3D_SRC")
if not SHARP3D_SRC:
    raise SystemExit("set SHARP3D_SRC to the sharp3d/src directory")
sys.path.insert(0, SHARP3D_SRC)

import numpy as np
import torch

tmp = Path(os.environ.get("TEMP", ".")) / "_sharp3d_dtype_check"
tmp.mkdir(parents=True, exist_ok=True)
onnx_path = tmp / "identity_f32.onnx"

# Build a minimal ONNX graph: Identity on a float32 input.
if not onnx_path.exists():
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3, 4, 4])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3, 4, 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(onnx_path))

import onnxruntime as ort

session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

# Minimal stand-in exposing exactly the attributes _forward_numpy touches.
import types

from sharp3d.ort_engine import ORTEncoder

stub = types.SimpleNamespace(
    _session=session,
    _input_name=session.get_inputs()[0].name,
    device=torch.device("cpu"),
    intermediate_features_ids=[5, 11, 17, 23],
)

# --- Case 1: float16 input (the FP16 pipeline case that used to raise) ---
x_fp16 = torch.randn(2, 3, 4, 4, dtype=torch.float16)
try:
    feats, inter = ORTEncoder._forward_numpy(stub, x_fp16)
except Exception as exc:
    print(f"FAIL: float16 input raised {type(exc).__name__}: {exc}")
    raise SystemExit(1)

assert feats.dtype == torch.float32, f"expected float32, got {feats.dtype}"
assert feats.shape == (2, 3, 4, 4), feats.shape
print(f"PASS float16 input accepted -> features {tuple(feats.shape)} {feats.dtype}")

# --- Case 2: float32 input still works ---
x_fp32 = torch.randn(2, 3, 4, 4, dtype=torch.float32)
feats32, _ = ORTEncoder._forward_numpy(stub, x_fp32)
assert feats32.dtype == torch.float32
print(f"PASS float32 input accepted -> features {tuple(feats32.shape)} {feats32.dtype}")

# --- Case 3: numerical sanity, fp16 input == fp32 of the same values ---
x = torch.randn(1, 3, 4, 4)
a, _ = ORTEncoder._forward_numpy(stub, x.half())
b, _ = ORTEncoder._forward_numpy(stub, x.float())
max_err = float((a - b).abs().max())
assert max_err < 1e-3, f"fp16/fp32 paths diverged: {max_err}"
print(f"PASS fp16 vs fp32 max abs diff {max_err:.2e}")

print("\nALL CHECKS PASSED")
