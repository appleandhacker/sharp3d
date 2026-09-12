"""Render a 2.5D parallax (swipe) video from the original ml-sharp ply.

Uses ONLY the original ml-sharp APIs (sharp.utils.camera + sharp.cli.render),
with the trajectory type the CLI does not expose.
"""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\yhm\.qoderworkcn\workspace\mrsw6dewe12d0mgy\ml-sharp\src")

import torch
from sharp.utils import camera
from sharp.utils.gaussians import load_ply
from sharp.cli.render import render_gaussians

ply = Path(r"C:\Users\yhm\.qoderworkcn\workspace\mrsw6dewe12d0mgy\outputs\_orig_sharp_test\a624182713728ee2b05eb2d6c400292e3506595.ply")
out = Path(r"C:\Users\yhm\.qoderworkcn\workspace\mrsw6dewe12d0mgy\outputs\_orig_sharp_test\a624182713728ee2b05eb2d6c400292e3506595.swipe.mp4")

gaussians, metadata = load_ply(ply)
params = camera.TrajectoryParams(
    type="swipe",          # 左右横扫 —— 最典型的 2.5D 视差
    max_disparity=0.08,
    max_zoom=0.15,
    num_steps=60,
    num_repeats=1,
)
render_gaussians(gaussians=gaussians, metadata=metadata,
                 output_path=out, params=params)
print("SWIPE VIDEO OK:", out, out.stat().st_size // 1024, "KB")
