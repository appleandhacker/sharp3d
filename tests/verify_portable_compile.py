"""End-to-end verification of the portable package with clean compile caches.

Simulates the target machine: base-runtime python + PYTHONPATH site-packages,
CC pinned to bundled TinyCC, empty inductor/triton caches (first-run state).
Run with the PACKAGE's runtime python.exe, not the dev venv.
"""
import os
import sys
import time

ROOT = r"C:\Users\yhm\Desktop\sharp3d"
# Fresh cache dirs (first-run state on the target machine)
CACHE = r"C:\Users\yhm\Desktop\sharp3d\sharp3d\.cache"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE + r"\inductor_test"
os.environ["TRITON_CACHE_DIR"] = CACHE + r"\triton_test"

sys.path.insert(0, ROOT + r"\sharp3d\src")
sys.path.insert(0, ROOT + r"\sharp-src")
sys.path.insert(0, ROOT + r"\env\Lib\site-packages")

import numpy as np
import torch

print("python:", sys.version.split()[0], flush=True)
print("import torch OK", torch.__version__, flush=True)

from sharp3d.predict import SharpPredictor
from sharp3d.unproject import prepare_input

t0 = time.time()
sp = SharpPredictor(device=torch.device("cuda"))
print(f"predictor loaded in {time.time()-t0:.0f}s", flush=True)

img = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
img_r, df, _ir, _sz = prepare_input(img, 1920 * 1.2, torch.device("cuda"))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    t1 = time.time()
    g = sp.predict(img_r, df)
    torch.cuda.synchronize()
print(f"first predict: {time.time()-t1:.1f}s (includes dynamo+triton+tcc compile)", flush=True)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    g = sp.predict(img_r, df)
torch.cuda.synchronize()
print("PREDICT OK:", type(g).__name__, flush=True)
