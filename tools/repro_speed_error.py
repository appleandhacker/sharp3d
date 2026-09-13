"""复现 GUI「速度优先」报错：与 sbs_tab._build_opts 完全相同的选项驱动 worker。

用法: python tools/repro_speed_error.py [quality|speed]
打印 worker 的全部信号响应（error / convert_progress / convert_done），
error 响应携带真实异常文本 —— GUI 上显示的就是它。
"""
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(os.environ.get("SHARP3D_REPO_ROOT",
                           Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "src"))
for _p in os.environ.get("SHARP3D_EXTRA_PATHS", "").split(";"):
    if _p:
        sys.path.insert(0, _p)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
print(f"[repro] 代码来源: {REPO / 'src'}", flush=True)

MODE = sys.argv[1] if len(sys.argv) > 1 else "speed"
TMP = Path(os.environ.get("TEMP", "/tmp")) / "sharp3d_speed_repro"
SRC = TMP / "src_640.mp4"
OUT = TMP / f"out_{MODE}.mp4"


def make_source():
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=12:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-shortest",
         str(SRC)], check=True, creationflags=0x08000000)


def main():
    make_source()
    if OUT.exists():
        OUT.unlink()

    import sharp3d.gui.worker as W
    responses = []

    def respond(name, args):
        responses.append((name, args))
        if name == "convert_progress":
            frame, total, fps, elapsed = args
            print(f"[progress] {frame}/{total} {fps:.2f}fps", flush=True)
        elif name == "error":
            print(f"[ERROR] {args}", flush=True)
        elif name == "convert_done":
            print(f"[done] {args}", flush=True)
        else:
            print(f"[{name}] {str(args)[:120]}", flush=True)

    worker = W._PipelineWorker(respond, threading.Event())

    opts = {
        "input": str(SRC), "output": str(OUT),
        "format": "full_sbs", "ipd_mm": 63.0, "convergence": 0,
        "strength": 1.0, "codec": "av1", "crf": 26,
        "audio": True, "decompose": "analytical",
        "depth": False, "ply": False, "edge_soften": False,
        "hdr_output": False, "perf_mode": MODE, "focal_35mm": None,
        "renderer": "higs", "out_fps": None,
        "out_scale": 1.0, "out_width": None,
        "temporal_stabilize": "off", "keyframe_interval": 1,
    }

    err = []

    def run():
        try:
            worker.convert(opts)
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            traceback.print_exc()

    t0 = time.time()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=900)
    print(f"\nelapsed={time.time()-t0:.0f}s thread_alive={t.is_alive()}")
    names = [n for n, _ in responses]
    if err:
        print(f"THREAD_EXCEPTION: {err[-0:]}" if False else
              f"THREAD_EXCEPTION: {err[0]!r}")
    if "error" in names:
        print(f"\nRESULT: REPRODUCED — error 响应: "
              f"{[a for n, a in responses if n == 'error']}")
        return 2
    if "convert_done" not in names:
        print("\nRESULT: NO_DONE（转换未完成，见上方线程异常/输出）")
        return 3
    print("\nRESULT: NO_ERROR（speed 模式全链路正常完成）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
