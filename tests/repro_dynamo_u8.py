"""Reproduce the target machine's UnicodeDecodeError under torch._dynamo.

Simulated conditions (from the user's target-machine log):
- Chinese characters in the cache paths (D:\桌面\...  ->  Desktop\测试缓存\...)
- No CUDA Toolkit: CUDA_PATH/CUDA_HOME removed, CUDA dirs stripped from PATH
- Empty inductor/triton caches (no prebuilt kernels)
"""
import os
import sys

CACHE = r"C:\Users\yhm\Desktop\测试缓存"
os.environ["TRITON_CACHE_DIR"] = CACHE + r"\triton"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE + r"\inductor"
os.environ.pop("CUDA_PATH", None)
os.environ.pop("CUDA_HOME", None)
os.environ["PATH"] = ";".join(
    p for p in os.environ["PATH"].split(";") if "cuda" not in p.lower()
)

sys.path.insert(0, r"C:\Users\yhm\.qoderworkcn\workspace\mrsw6dewe12d0mgy\sharp3d\src")

import numpy as np
import torch

print("step1: imports", flush=True)
from sharp3d.predict import SharpPredictor
from sharp3d.unproject import prepare_input

print("step2: loading predictor (quality/compile)...", flush=True)
sp = SharpPredictor(device=torch.device("cuda"))

print("step3: first predict (triggers dynamo+triton compile)...", flush=True)
img = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
img_r, df, _ir, _sz = prepare_input(img, 1920 * 1.2, torch.device("cuda"))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    g = sp.predict(img_r, df)
torch.cuda.synchronize()
print("PREDICT OK:", type(g).__name__, flush=True)
