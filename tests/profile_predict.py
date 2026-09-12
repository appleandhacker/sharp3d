"""Kernel-level breakdown of one SHARP predict call.

Splits the ~410ms predict into: ORT encoder time vs SPN/composer kernels,
and ranks the heaviest CUDA kernels so the right optimization lever
(TRT export vs CUDA graphs vs kernel tuning) can be picked on evidence.

    sharp3d-env\\Scripts\\python.exe tests\\profile_predict.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from sharp3d.predict import SharpPredictor
from sharp3d.unproject import prepare_input, INTERNAL_SHAPE
import numpy as np

dev = torch.device("cuda")
print("loading predictor (warm cache)…", flush=True)
t0 = time.time()
sp = SharpPredictor(device=dev)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

# 合成输入（真实图片更好，但 kernel 分布对内容不敏感）
img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
img_r, df, ir, _ = prepare_input(img, 1920 * 1.2, dev)

# 预热（编译缓存命中 + 首次推理）
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for _ in range(3):
        sp.predict(img_r, df)
torch.cuda.synchronize()

# 计时基线（无 profiler 开销）
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5):
        sp.predict(img_r, df)
    torch.cuda.synchronize()
print(f"predict 均值 (无 profiler): {(time.time()-t0)/5*1000:.1f}ms\n", flush=True)

# profiler: 剖析一次 predict
from torch.profiler import profile, ProfilerActivity
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        sp.predict(img_r, df)
        torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=28,
    max_name_column_width=70), flush=True)

# ORT 部分：单独计时（profiling.add_ort 已在 ORTEncoder 里累计）
from sharp3d import profiling
print(f"\nORT 编码器累计 (label): {profiling.ort_ms}", flush=True)
