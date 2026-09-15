#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从断点继续一次被中断的视频转换（复用已完成的分片 tmp 文件）。

场景：转换被外部打断（断电/休眠/被杀）后，输出目录里留下
`<名字>.tmp.mp4`（分片 MP4，可边写边播）。本工具只转换剩余帧，产出
`<名字>.part2.mp4`，再用 concat demuxer 无损拼接成完整成品。

用法示例：
    python tools/resume_convert.py \\
        --input  第5段源.mp4 \\
        --output 第5段_sbs.mp4 \\
        --tmp    第5段_sbs.tmp.mp4 \\
        --resume-frames 50780 --resume-seek 1694.526125 \\
        --focal 135 --strength 1.2

关键点（都有实测依据，别随意改）：
  * --resume-frames 必须等于已完成 tmp 的**视频帧数**（用
    tools/probe_frag.py 解析分片采样表，或 ffprobe -count_packets）。
  * --resume-seek 是第 resume-frames 帧在**源文件里的绝对时间**。源若为
    轻微 VFR，用帧数/fps 推算会有 ±1~5 帧偏差，必须用 PTS 实测：
      ffmpeg -ss <粗位置> -copyts -i SRC -vf "select=gte(t\\,X),showinfo" -frames:v 8 -f null -
    再把不同 X 阈值的总帧数反推索引（count = 总帧数 − 首帧索引）。
    不给 --resume-seek 时退回近似时间 seek（接缝可能重复/丢 1 帧）。
  * 音频输入会同步做 -ss；缺了它续转段的音轨会从源 0 秒开始，与画面错位。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main() -> int:
    ap = argparse.ArgumentParser(description="从断点继续视频转换")
    ap.add_argument("--input", required=True, help="源视频")
    ap.add_argument("--output", required=True, help="最终成品路径")
    ap.add_argument("--tmp", required=True, help="中断留下的分片 tmp（part1）")
    ap.add_argument("--resume-frames", type=int, required=True,
                    help="part1 已完成的视频帧数")
    ap.add_argument("--resume-seek", type=float, default=None,
                    help="第 resume-frames 帧的绝对源时间（秒，强烈建议给）")
    ap.add_argument("--focal", type=float, default=None, help="35mm 等效焦距")
    ap.add_argument("--strength", type=float, default=1.0, help="立体强度")
    ap.add_argument("--ipd-mm", type=float, default=63.0)
    ap.add_argument("--convergence", type=float, default=0.0, help="0=自动")
    ap.add_argument("--format", default="full_sbs", help="full_sbs/half_sbs/...")
    ap.add_argument("--codec", default="h264", choices=["h264", "h265", "av1"])
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--perf-mode", default="quality",
                    choices=["quality", "speed"])
    ap.add_argument("--renderer", default="standard",
                    choices=["standard", "higs"])
    ap.add_argument("--decompose", default="analytical",
                    choices=["analytical", "svd"])
    ap.add_argument("--keyframe-interval", type=int, default=1)
    ap.add_argument("--out-scale", type=float, default=1.0)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-concat", action="store_true",
                    help="只转剩余帧，不拼接到 --tmp 上")
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    args = ap.parse_args()

    src, out, part1 = Path(args.input), Path(args.output), Path(args.tmp)
    for p, name in ((src, "源"), (part1, "part1(tmp)")):
        if not p.exists():
            print(f"!! {name} 不存在: {p}", file=sys.stderr)
            return 2
    if out.exists() and not args.yes:
        print(f"!! 成品已存在: {out}（加 --yes 覆盖）", file=sys.stderr)
        return 2

    part2 = out.with_suffix(".part2.mp4")
    opts = {
        "input": str(src), "output": str(part2),
        "format": args.format, "ipd_mm": args.ipd_mm,
        "convergence": args.convergence, "strength": args.strength,
        "codec": args.codec, "crf": args.crf, "audio": not args.no_audio,
        "hdr_output": False, "decompose": args.decompose,
        "depth": False, "ply": False, "edge_soften": False,
        "perf_mode": args.perf_mode, "focal_35mm": args.focal,
        "renderer": args.renderer, "out_fps": None,
        "out_scale": args.out_scale, "out_width": None,
        "temporal_stabilize": "off",
        "keyframe_interval": args.keyframe_interval,
        "resume_frames": args.resume_frames,
    }
    if args.resume_seek is not None:
        opts["resume_seek"] = args.resume_seek

    cancel = threading.Event()
    last = [0.0]

    def respond(name, payload):
        if name == "convert_progress":
            done, total, fps, elapsed = payload
            if time.time() - last[0] >= 30:
                last[0] = time.time()
                eta = (total - done) / fps / 60 if fps > 0 else 0
                print(f"  {done}/{total} ({done / max(total, 1) * 100:.1f}%)  "
                      f"{fps:.2f} fps  剩余约 {eta:.0f} min", flush=True)
        elif name in ("error", "convert_done"):
            print(f"[{name}] {payload}", flush=True)
        elif name == "status":
            print(f"[status] {payload[0] if payload else ''}", flush=True)

    import sharp3d.gui.worker as W  # noqa: E402
    worker = W._PipelineWorker(respond, cancel)
    err: list = []

    def run():
        try:
            worker.convert(opts)
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            import traceback
            traceback.print_exc()

    print(f"续转: {src.name} → {part2.name}  "
          f"(从第 {args.resume_frames} 帧起)", flush=True)
    t = threading.Thread(target=run, daemon=True)
    t0 = time.time()
    t.start()
    while t.is_alive():
        t.join(timeout=15)
    print(f"续转结束，用时 {(time.time() - t0) / 60:.1f} min", flush=True)
    if err:
        print(f"!! 失败: {err[-1]!r}", file=sys.stderr)
        return 1
    if not part2.exists():
        print("!! 未产出 part2", file=sys.stderr)
        return 1
    if args.no_concat:
        print(f"完成（未拼接）: {part2}")
        return 0

    import subprocess  # noqa: E402
    lst = part2.with_suffix(".concat.txt")
    lst.write_text("ffconcat version 1.0\n"
                   f"file '{part1.as_posix()}'\n"
                   f"file '{part2.as_posix()}'\n", encoding="utf-8")
    print(f"拼接 {part1.name} + {part2.name} → {out.name}", flush=True)
    r = subprocess.run(["ffmpeg", "-y", "-v", "warning", "-f", "concat",
                        "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-map", "0:v:0", "-map", "0:a:0?", str(out)])
    if r.returncode != 0:
        print("!! 拼接失败:\n" + r.stderr.decode("utf-8", "replace")[-2000:],
              file=sys.stderr)
        return 1
    print(f"完成: {out}  {out.stat().st_size / 2**30:.2f} GiB")
    print("（part1/part2 未删除，确认后可自行清理）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
