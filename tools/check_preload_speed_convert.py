"""回归：预加载（画质）→ 未完成时排队 speed 转换 → 进度归属不得错乱。

镜像用户操作序列：GUI 启动即 preload（quality，job1）；用户选速度并点
开始转换（job2 排在 preload 之后）。model_ready 在两次构建时都会发出
（非终结），转换的 progress/done 必须全程归属 job2。
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SHARP3D_NO_COMPILE", "1")

from PySide6.QtCore import QCoreApplication  # noqa: E402


def main():
    app = QCoreApplication([])
    import sharp3d.gui.worker as W

    TMP = Path(os.environ.get("TEMP", "/tmp")) / "sharp3d_mode_switch_test"
    src = TMP / "src.mp4"
    assert src.exists(), "先运行 check_mode_switch_progress.py 生成素材"

    engine = W.EngineProcess()
    got = {"progress": [], "done": [], "error": [], "model_ready": 0}

    def on_progress(f, t, fps, el, jid=-1):
        got["progress"].append((f, t, jid))

    def on_done(r):
        got["done"].append(dict(r))
        print(f"[done] job_id={r.get('job_id')} cancelled={r.get('cancelled')}",
              flush=True)

    def on_error(msg, jid=-1):
        got["error"].append((msg, jid))
        print(f"[ERROR] {msg}", flush=True)

    def on_model_ready():
        got["model_ready"] += 1
        print("[model_ready]", flush=True)

    engine.convert_progress.connect(on_progress)
    engine.convert_done.connect(on_done)
    engine.error.connect(on_error)
    engine.model_ready.connect(on_model_ready)

    # 1) 启动预加载（与 main_window 一致）
    engine.preload()
    time.sleep(0.3)
    # 2) 用户在预加载未完成时选速度并点开始（convert 排队在其后）
    jid = engine.new_job()
    opts = {"input": str(src), "output": str(TMP / "out_preload.mp4"),
            "format": "full_sbs", "ipd_mm": 63.0, "convergence": 0,
            "strength": 1.0, "codec": "h264", "crf": 30, "audio": True,
            "decompose": "analytical", "depth": False, "ply": False,
            "edge_soften": False, "hdr_output": False, "perf_mode": "speed",
            "focal_35mm": None, "renderer": "higs", "out_fps": None,
            "out_scale": 1.0, "out_width": None, "temporal_stabilize": "off",
            "keyframe_interval": 1}
    engine.convert(opts, jid)

    deadline = time.time() + 600
    while time.time() < deadline:
        app.processEvents()
        if got["done"] or got["error"]:
            for _ in range(5):
                app.processEvents()
                time.sleep(0.05)
            break
        time.sleep(0.05)
    try:
        engine.stop()
    except Exception:  # noqa: BLE001
        pass

    n_my = sum(1 for _, _, j in got["progress"] if j == jid)
    done_ok = bool(got["done"]) and got["done"][0].get("job_id") == jid \
        and not got["done"][0].get("cancelled")
    ok = n_my > 0 and done_ok and not got["error"]
    # 中止语义：被取代的画质构建不得完成（model_ready 只应出现 1 次——
    # 速度构建；旧行为是 2 次：画质预加载跑完 + 速度重建）
    print(f"model_ready: {got['model_ready']} 次（中止语义要求 =1）")
    if got["model_ready"] != 1:
        print("FAIL: model_ready 次数 != 1 —— 被取代的构建没有被中止，"
              "或速度构建未完成")
        ok = False
    print(f"progress: {len(got['progress'])} 条, 归属正确(jid={jid}): {n_my}")
    print(f"done: {bool(got['done'])} 归属正确: {done_ok}")
    print(f"error: {got['error'][:1]}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
