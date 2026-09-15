# -*- coding: utf-8 -*-
"""接缝精确比对：part1 末帧 / part2 首帧 是否原样出现在成品的接缝处。

这才是真正的验收 —— 数量对得上不代表接缝对得上（可能重复或跳帧）。
"""
import subprocess
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FFMPEG = "ffmpeg"
H = r"C:\Users\yhm\Desktop\hxytemp"
PART1 = H + r"\第5段_02.08.30-02.40.37_sbs.tmp.mp4"
PART2 = H + r"\第5段_02.08.30-02.40.37_sbs.part2.mp4"
FINAL = H + r"\第5段_02.08.30-02.40.37_sbs.mp4"
W, HH = 7680, 2160
FS = W * HH * 3


def grab(path, t, n):
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-hwaccel", "cuda",
           "-ss", f"{t:.3f}", "-i", path, "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(cmd, capture_output=True).stdout
    k = len(buf) // FS
    return [np.frombuffer(buf[i * FS:(i + 1) * FS], dtype=np.uint8).reshape(HH, W, 3)
            for i in range(k)]


def mad(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# part1 末帧：从 1694.40 起取 8 帧，最后一帧即 part1 的收尾帧
p1 = grab(PART1, 1694.40, 8)
last_p1 = p1[-1]
print(f"part1 尾窗取到 {len(p1)} 帧；末帧签名 MAD(倒数第二, 末帧)="
      f"{mad(p1[-2], p1[-1]):.3f}")

# part2 首帧
p2 = grab(PART2, 0.0, 3)
first_p2 = p2[0]
print(f"part2 首窗取到 {len(p2)} 帧；MAD(首帧, 次帧)={mad(p2[0], p2[1]):.3f}")

# 成品接缝窗（成品时间轴里接缝位于 ~1694.528s）
fin = grab(FINAL, 1694.40, 12)
print(f"成品接缝窗取到 {len(fin)} 帧")
print("\n 序号 | MAD(成品[i], part1末帧) | MAD(成品[i], part2首帧) | 相邻MAD")
best = None
for i, f in enumerate(fin):
    a = mad(f, last_p1)
    b = mad(f, first_p2)
    nxt = mad(f, fin[i + 1]) if i + 1 < len(fin) else float("nan")
    print(f"  {i:>4} | {a:>22.4f} | {b:>22.4f} | {nxt:>7.3f}")
    if a < 0.001:
        best = i
if best is not None:
    print(f"\n→ 成品第 {best} 帧与 part1 末帧逐像素相同")
    if best + 1 < len(fin):
        print(f"→ 成品第 {best+1} 帧与 part2 首帧的 MAD = "
              f"{mad(fin[best+1], first_p2):.4f}")
