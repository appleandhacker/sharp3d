"""Performance benchmark for the sharp3d conversion pipeline.

Drives the *production* GUI worker path (_PipelineWorker, Qt-free) so the
numbers include the real decode/encode threads, pinned-buffer D2H and all
pipeline overlap — not an idealized reimplementation.

    sharp3d-env\\Scripts\\python.exe tests\\bench.py --frames 24

Requires SHARP3D_PROFILE=1 (set automatically) for per-stage timings:
    [PROF] lines: predict / stabilize / render+pack (CUDA events) +
                  d2h+write (pipeline wall) + ort_enc (encoder wall)
    [PROF][sbs] encode thread summary at the end of each run.
"""
import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--out", default=str(ROOT.parent / "outputs"))
ap.add_argument("--only", default=None, help="只跑指定 label 的变体")
args = ap.parse_args()

os.environ.setdefault("SHARP3D_PROFILE", "1")  # =2 enables per-frame output

import torch  # noqa: E402
from sharp3d.gui.worker import _PipelineWorker  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

_done: list[dict] = []


def respond(name, args_):
    if name == "convert_done":
        _done.append(args_[0])
    elif name == "error":
        print(f"  [ERROR] {args_[0]}", flush=True)


cancel = mp.Event()
worker = _PipelineWorker(respond, cancel)

# ── model load timing (first convert pays it) ───────────────────────────
print("\n###MODEL_LOAD###", flush=True)
_t0 = time.time()
worker._ensure_pipeline()
torch.cuda.synchronize()
print(f"###MODEL_LOAD_DONE### {time.time() - _t0:.1f}s", flush=True)
print(f"vram after load: alloc={torch.cuda.memory_allocated()/2**20:.0f}MB "
      f"reserved={torch.cuda.memory_reserved()/2**20:.0f}MB", flush=True)

VARIANTS = [
    # (label, source, renderer, kf, int8, perf_mode)
    ("1080p_fullsbs_standard", "_bench_1080p.mp4", "standard", 1, False, "quality"),
    ("1080p_fullsbs_higs",     "_bench_1080p.mp4", "higs",     1, False, "quality"),
    ("4k_fullsbs_standard",    "_bench_4k.mp4",    "standard", 1, False, "quality"),
    ("4k_fullsbs_higs",        "_bench_4k.mp4",    "higs",     1, False, "quality"),
    ("4k_fullsbs_higs_kf2",    "_bench_4k.mp4",    "higs",     2, False, "quality"),
    ("4k_fullsbs_higs_kf3",    "_bench_4k.mp4",    "higs",     3, False, "quality"),
    ("4k_speed21_higs",        "_bench_4k.mp4",    "higs",     1, False, "speed"),
    ("4k_fullsbs_higs_int8",   "_bench_4k.mp4",    "higs",     1, True,  "quality"),
]

summary = []
for label, src, renderer, kf, int8, perf_mode in VARIANTS:
    if args.only and label != args.only:
        continue
    inp = OUT / src
    if not inp.exists():
        print(f"SKIP {label}: {inp} 不存在", flush=True)
        continue
    out = OUT / f"_bench_out_{label}.mp4"
    if out.exists():
        out.unlink()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    print(f"\n{'=' * 74}\n###BENCH### {label}  (src={src}, renderer={renderer}, "
          f"kf={kf}, int8={int8}, perf={perf_mode})\n{'=' * 74}", flush=True)
    t0 = time.time()
    worker.convert({
        "input": str(inp), "output": str(out),
        "ipd_mm": 63.0, "strength": 1.0, "convergence": 0.0,
        "decompose": "analytical", "codec": "h264", "crf": 20,
        "audio": False, "out_fps": None, "out_scale": 1.0,
        "out_width": None, "format": "full_sbs",
        "temporal_stabilize": "adaptive", "keyframe_interval": kf,
        "edge_soften": False, "depth": False, "ply": False,
        "perf_mode": perf_mode, "renderer": renderer, "int8": int8,
    })
    wall = time.time() - t0
    r = _done[-1] if _done else {}
    peak = torch.cuda.max_memory_allocated() / 2**20
    entry = {
        "label": label, "wall_s": round(wall, 2),
        "n_frames": r.get("n_frames"), "fps": round(r.get("fps", 0.0), 3),
        "size": list(r.get("size", [])),
        "peak_vram_mb": round(peak),
    }
    summary.append(entry)
    print(f"###RESULT### {json.dumps(entry)}", flush=True)

print("\n" + "=" * 74)
print("###SUMMARY###")
print(json.dumps(summary, indent=2, ensure_ascii=False))
(OUT / "_bench_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved:", OUT / "_bench_summary.json")
