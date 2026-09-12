"""Convert one clip through a single pipeline mode (subprocess-isolated).

Usage: python tests/_convert_one.py <mode> <int8 0|1>
Reads env overrides (SHARP3D_FULL_TRT etc.) from the parent.
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

mode = sys.argv[1]
int8 = len(sys.argv) > 2 and sys.argv[2] == "1"
SRC = (Path(sys.argv[3]) if len(sys.argv) > 3
       else ROOT.parent / "outputs" / "_bench_4k.mp4")

OUT = ROOT.parent / "outputs" / f"_cmp_{mode}.mp4"

from sharp3d.gui.worker import _PipelineWorker  # noqa: E402

cancel = mp.Event()
worker = _PipelineWorker(lambda n, a: None, cancel)
if OUT.exists():
    OUT.unlink()

t0 = time.time()
worker.convert({
    "input": str(SRC), "output": str(OUT),
    "ipd_mm": 63.0, "strength": 1.0, "convergence": 0.0,
    "decompose": "analytical", "codec": "h264", "crf": 20,
    "audio": False, "out_fps": None, "out_scale": 1.0,
    "out_width": None, "format": "full_sbs",
    "keyframe_interval": 1,
    "edge_soften": False, "depth": False, "ply": False,
    "perf_mode": "quality", "renderer": "higs", "int8": int8,
})
wall = time.time() - t0
n_frames = 24
print(f"###TIME### {{\"mode\": \"{mode}\", \"wall_s\": {wall:.2f}, "
      f"\"fps\": {n_frames / wall:.3f}, \"out\": \"{OUT}\"}}", flush=True)
assert OUT.exists(), f"{mode} 转换失败"
