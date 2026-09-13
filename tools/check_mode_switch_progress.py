"""回归：速度模式转换中（管线重建发 model_ready）不得打断进度归属。

事故（2026-09-13）：model_ready 在 _TERMINAL_RESPONSES 里，而 speed 优先的
首次转换会在任务中途重建管线并发出 model_ready —— _poll 据此把转换任务
自己的 job_id 弹出队列，后续 convert_progress 全部被打成 job_id=-1、被
tab 的归属过滤丢弃：GUI 冻结在上次状态，转换在后台正常跑完。

本测试走真实 EngineProcess（父子进程 + QTimer 轮询 + job 归属层）：
fresh worker 的首个 convert(speed) 必然触发 mid-job 管线构建，正好复现
旧代码的触发条件。断言：
  1. 收到的 convert_progress / convert_done 都带本任务 job_id（非 -1）
  2. 至少一条 convert_progress，且 convert_done 正常到达
  3. 全程无 error
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TMP = Path(os.environ.get("TEMP", "/tmp")) / "sharp3d_mode_switch_test"
SRC = TMP / "src.mp4"
OUT = TMP / "out.mp4"


def make_source():
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=12:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-shortest",
         str(SRC)], check=True, creationflags=0x08000000)


def main():
    from PySide6.QtCore import QCoreApplication, QTimer

    app = QCoreApplication.instance() or QCoreApplication([])
    os.environ.setdefault("SHARP3D_NO_COMPILE", "1")  # 测试提速；与 bug 无关
    import sharp3d.gui.worker as W

    engine = W.EngineProcess()
    got = {"progress": [], "done": [], "error": []}
    job_id_holder = {}

    def on_progress(frame, total, fps, elapsed, job_id=-1):
        got["progress"].append((frame, total, job_id))

    def on_done(result):
        got["done"].append(dict(result))

    def on_error(msg, job_id=-1):
        got["error"].append((msg, job_id))

    engine.convert_progress.connect(on_progress)
    engine.convert_done.connect(on_done)
    engine.error.connect(on_error)

    make_source()
    OUT.unlink(missing_ok=True)

    opts = {
        "input": str(SRC), "output": str(OUT),
        "format": "full_sbs", "ipd_mm": 63.0, "convergence": 0,
        "strength": 1.0, "codec": "h264", "crf": 30, "audio": True,
        "decompose": "analytical", "depth": False, "ply": False,
        "edge_soften": False, "hdr_output": False, "perf_mode": "speed",
        "focal_35mm": None, "renderer": "higs", "out_fps": None,
        "out_scale": 1.0, "out_width": None, "temporal_stabilize": "off",
        "keyframe_interval": 1,
    }
    job_id_holder["id"] = engine.new_job()
    engine.convert(opts, job_id_holder["id"])
    my_id = job_id_holder["id"]

    deadline = time.time() + 600
    while time.time() < deadline:
        app.processEvents()
        if got["done"] or got["error"]:
            # 排空队列中可能还排着的信号
            for _ in range(5):
                app.processEvents()
                time.sleep(0.05)
            break
        time.sleep(0.05)

    try:
        engine.stop()
    except Exception:  # noqa: BLE001
        pass

    ok = True
    if got["error"]:
        print(f"FAIL: 收到 error: {got['error']}")
        ok = False
    if not got["done"]:
        print("FAIL: 未收到 convert_done（超时）")
        ok = False
    else:
        d = got["done"][0]
        if d.get("job_id") != my_id:
            print(f"FAIL: convert_done job_id={d.get('job_id')} != {my_id} "
                  f"（归属被 model_ready 弹出打断）")
            ok = False
        if d.get("cancelled"):
            print("FAIL: convert_done 标记为 cancelled")
            ok = False
    n_my = sum(1 for _, _, jid in got["progress"] if jid == my_id)
    if not n_my:
        print(f"FAIL: 没有任何 convert_progress 带本任务 job_id="
              f"{my_id}（事件被丢进 -1）：样本 "
              f"{got['progress'][:3]} …")
        ok = False

    print(f"\nprogress 事件: {len(got['progress'])} 条 "
          f"(归属正确 {n_my})，done={bool(got['done'])}, "
          f"error={got['error'][:1]}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
