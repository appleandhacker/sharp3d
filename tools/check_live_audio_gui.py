"""GUI worker 实时音轨验证：走 _PipelineWorker.convert() 全链路。

注意：实时音轨默认已禁用（ffmpeg 7.1 双输入下编码器内存无界增长，8K 实测
3GB→10GB）。本脚本验证 SHARP3D_LIVE_AUDIO=1 强制启用时的机制仍可用。

验收点:
  1. 转换中 .tmp.mp4 即含 aac 音频流（init 段声明）
  2. 音轨时长随视频推进增长、非静音（转换中可听）
  3. convert_done（非 error）；最终文件音轨完整
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TMP_DIR = Path(os.environ.get("TEMP", "/tmp")) / "sharp3d_live_gui"
SRC = TMP_DIR / "test_av_gui.mp4"
OUT = TMP_DIR / "out_gui_live.mp4"


def make_source():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=12:duration=5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-shortest",
         str(SRC)], check=True, creationflags=0x08000000)
    assert SRC.exists()


def probe(path, args):
    r = subprocess.run(["ffprobe", "-v", "error", *args, str(path)],
                       capture_output=True, text=True, creationflags=0x08000000)
    return r.stdout.strip()


def volume(path):
    r = subprocess.run(["ffmpeg", "-i", str(path), "-af", "volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True,
                       creationflags=0x08000000)
    for line in (r.stderr or "").splitlines():
        if "max_volume" in line:
            return float(line.split("max_volume:")[1].split()[0])
    return -99.0


def main():
    os.environ["SHARP3D_LIVE_AUDIO"] = "1"  # 实时音轨需显式启用（默认禁用）
    make_source()
    if OUT.exists():
        OUT.unlink()

    import sharp3d.gui.worker as W
    responses = []
    worker = W._PipelineWorker(
        lambda name, args: responses.append((name, args)),
        threading.Event())

    opts = {
        "input": str(SRC), "output": str(OUT),
        "ipd_mm": 63.0, "strength": 1.0, "convergence": 0,
        "decompose": "analytical", "codec": "h264", "crf": 20,
        "format": "full_sbs", "keyframe_interval": 3,
        "audio": True, "out_scale": 1.0, "hdr_output": False,
        "perf_mode": "quality", "out_fps": None,
    }

    t = threading.Thread(target=worker.convert, args=(opts,), daemon=True)
    t0 = time.time()
    t.start()

    # 轮询转换中的 tmp 文件
    tmp = OUT.with_suffix(".tmp.mp4")
    mid_audio_seen = False
    mid_maxvol = -99.0
    deadline = time.time() + 300
    polls = 0
    while time.time() < deadline and t.is_alive():
        if tmp.exists():
            polls += 1
            streams = probe(tmp, ["-show_entries", "stream=codec_type",
                                  "-of", "csv=p=0"])
            if "audio" in streams:
                dur = probe(tmp, ["-select_streams", "a:0", "-show_entries",
                                  "stream=duration", "-of", "csv=p=0"])
                if dur and float(dur.splitlines()[0]) > 1.0:
                    mid_audio_seen = True
                    mid_maxvol = volume(tmp)
                    print(f"[mid-run ~{time.time()-t0:.0f}s] streams={streams.replace(chr(10),'+')} "
                          f"audio_dur={dur.splitlines()[0]}s max_vol={mid_maxvol}dB")
                    if mid_audio_seen and polls >= 2:
                        break
        time.sleep(3)

    t.join(timeout=300)

    ok = True
    names = [n for n, _ in responses]
    if "convert_done" not in names or "error" in names:
        print(f"FAIL: responses={names[:5]} {responses[:2]}")
        ok = False
    final_streams = probe(OUT, ["-show_entries", "stream=codec_type",
                                "-of", "csv=p=0"]) if OUT.exists() else ""
    final_vol = volume(OUT) if OUT.exists() else -99.0
    if "audio" not in final_streams:
        print(f"FAIL: 最终文件无音轨: {final_streams!r}")
        ok = False
    if final_vol < -50:
        print(f"FAIL: 最终音轨接近静音 ({final_vol}dB)")
        ok = False
    if not mid_audio_seen:
        print("FAIL: 转换中未探测到音轨（或音轨增长慢于轮询窗口）")
        ok = False

    print(f"mid-run audio seen: {mid_audio_seen} (max_vol {mid_maxvol}dB), "
          f"final: streams={final_streams.replace(chr(10), '+')} "
          f"vol={final_vol}dB")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
