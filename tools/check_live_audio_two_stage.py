"""实时音轨（两段式：编码 | 混流）验证。

背景：单进程同时做视频编码 + 音频编码/mux 时，ffmpeg 在 8K 下稳定占住约
7GB（1.85GB→8.93GB）。改成两段式（ffmpeg#1 编码 → ivf/mpegts 裸流管道 →
ffmpeg#2 与音频文件混流）后峰值 1.88GB。本脚本验证改造后的 writer：
  01 默认即启用实时音轨（无需环境变量），转换中 tmp 已含音频
  02 SHARP3D_LIVE_AUDIO=0 可退回单进程 + close 时复用
  03 8K 下两段式内存不出现 7GB 平台
  04 abort() 能收掉两个进程并删除 tmp
  05 HDR（Hdr10Writer）两段式：音频 + hvc1 + colr 元数据齐全
"""
import ctypes
import os
import struct
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FFMPEG, FFPROBE = "ffmpeg", "ffprobe"
TMP = Path(os.environ.get("TEMP", "/tmp")) / "sharp3d_two_stage"
TMP.mkdir(parents=True, exist_ok=True)
AUDIO = TMP / "audio5.wav"
AUDIO_LONG = TMP / "audio120.wav"

RESULTS = []
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


_k32 = ctypes.windll.kernel32
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_psapi = ctypes.windll.psapi
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
_psapi.GetProcessMemoryInfo.argtypes = (
    wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD)


def ws_mb(pid):
    h = _k32.OpenProcess(0x1000, False, pid)
    if not h:
        return 0.0
    p = PMC()
    p.cb = ctypes.sizeof(p)
    ok = _psapi.GetProcessMemoryInfo(h, ctypes.byref(p), p.cb)
    _k32.CloseHandle(h)
    return p.WorkingSetSize / 1048576.0 if ok else 0.0


def pid_alive(pid):
    """用 tasklist 判定：ctypes 的 OpenProcess 在进程刚退出、PID 尚未回收时
    会误判为存活（实测 abort 明明已生效，却报"仍有进程存活"）。"""
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                       capture_output=True, text=True, errors="replace")
    return "ffmpeg" in r.stdout.lower()


def streams(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=codec_type", "-of", "csv=p=0", str(path)],
                       capture_output=True)
    return r.stdout.decode(errors="replace").split()


def is_fragmented(path):
    size = os.path.getsize(path)
    pos = 0
    with open(path, "rb") as f:
        while pos + 8 <= size:
            f.seek(pos)
            hdr = f.read(16)
            if len(hdr) < 8:
                break
            b = struct.unpack(">I", hdr[:4])[0]
            t = hdr[4:8].decode("latin1", "replace")
            hs = 8
            if b == 1:
                b = struct.unpack(">Q", hdr[8:16])[0]
                hs = 16
            elif b == 0:
                b = size - pos
            if t == "moov":
                f.seek(pos)
                return b"mvex" in f.read(b)
            pos += max(b, hs)
    return True


def gen_audio():
    for path, dur in ((AUDIO, 5), (AUDIO_LONG, 120)):
        if not path.exists():
            subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi",
                            "-i", f"sine=frequency=440:duration={dur}",
                            str(path)], capture_output=True)


# ------------------------------------------------------------------ checks

@check("01 默认启用实时音轨：双进程 + 转换中 tmp 已含音频")
def c01():
    from sharp3d.video import VideoWriter
    out = TMP / "vw_default.mp4"
    out.unlink(missing_ok=True)
    w = VideoWriter(out, fps=12, width=640, height=360, codec="h264", crf=30,
                    audio_source=AUDIO)
    assert w._mux_proc is not None, "默认应启用两段式（无环境变量）"
    assert w._mux_proc.pid != w._proc.pid, "应是两个进程"
    for i in range(60):
        w.append_frame(np.full((360, 640, 3), (i * 9) % 256, dtype=np.uint8))
    # 收尾前轮询：转换中的 tmp 应已含音频轨
    mids = []
    for _ in range(15):
        if w.tmp_path.exists():
            mids = streams(w.tmp_path)
            if "audio" in mids:
                break
        time.sleep(0.3)
    w.close()
    assert "audio" in (mids or []), f"转换中 tmp 应已含音频流，实际 {mids}"
    assert "audio" in streams(out), "成品应含音轨"
    assert not is_fragmented(out), "成品应是普通 faststart MP4"
    assert w._mux_proc.poll() is not None, "混流进程应已退出"


@check("02 SHARP3D_LIVE_AUDIO=0 退回单进程，成品仍有音轨")
def c02():
    code = (
        "import sys, numpy as np; sys.path.insert(0, r'%s');"
        "from pathlib import Path; from sharp3d.video import VideoWriter;"
        "out = Path(r'%s'); out.unlink(missing_ok=True);"
        "w = VideoWriter(out, fps=12, width=640, height=360, codec='h264',"
        " crf=30, audio_source=Path(r'%s'));"
        "assert w._mux_proc is None, 'LIVE_AUDIO=0 时不应有两段式';"
        "[w.append_frame(np.full((360,640,3), (i*9)%%256, dtype=np.uint8))"
        " for i in range(40)];"
        "w.close(); print('SINGLE_STAGE_OK')"
    ) % (ROOT / "src", TMP / "vw_off.mp4", AUDIO)
    env = dict(os.environ, SHARP3D_LIVE_AUDIO="0", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True)
    so, se = (r.stdout.decode(errors="replace"),
              r.stderr.decode(errors="replace"))
    assert "SINGLE_STAGE_OK" in so, f"应走单进程: {so[-200:]} {se[-400:]}"
    assert "audio" in streams(TMP / "vw_off.mp4"), "成品应含音轨（close 时复用）"


@check("03 8K 两段式：内存不出现 7GB 平台")
def c03():
    from sharp3d.video import VideoWriter
    out = TMP / "vw_8k.mp4"
    out.unlink(missing_ok=True)
    # 音频必须比视频长：否则 -shortest 会让混流进程提前退出，管道断裂
    w = VideoWriter(out, fps=30, width=7680, height=2160, codec="av1",
                    crf=30, audio_source=AUDIO_LONG)
    peaks = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            tot = sum(ws_mb(p.pid) for p in (w._proc, w._mux_proc)
                      if p is not None and p.poll() is None)
            peaks.append(tot)
            stop.wait(1.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    frame = np.empty((2160, 7680, 3), dtype=np.uint8)
    try:
        for i in range(1200):
            frame.fill((i * 7) % 256)
            w.append_frame(frame)
        w.close()
    finally:
        stop.set()
        th.join(timeout=2)
        # 中途失败也要收掉两个 ffmpeg，别把僵尸进程留给后面的检查
        if w._proc.poll() is None or (w._mux_proc is not None
                                      and w._mux_proc.poll() is None):
            w.abort()
    vals = [v for v in peaks if v > 0]
    peak = max(vals) if vals else 0
    base = vals[0] if vals else 0
    grow = peak - base
    print(f"      8K 两段式：首={base:.0f}MB 峰={peak:.0f}MB 增量={grow:+.0f}MB")
    # 单进程实测会涨到 +7GB 平台；两段式应保持在 2GB 以内
    assert grow < 2000, f"两段式内存增量应 <2GB，实测 {grow:+.0f}MB"


@check("04 abort() 收掉两个进程并删除 tmp")
def c04():
    from sharp3d.video import VideoWriter
    out = TMP / "vw_abort.mp4"
    out.unlink(missing_ok=True)
    w = VideoWriter(out, fps=12, width=640, height=360, codec="h264", crf=30,
                    audio_source=AUDIO)
    for i in range(5):
        w.append_frame(np.full((360, 640, 3), i * 40, dtype=np.uint8))
    pids = [w._proc.pid, w._mux_proc.pid]
    w.abort()
    time.sleep(0.5)
    assert not w.tmp_path.exists(), "abort 后不应残留 tmp"
    alive = [p for p in pids if pid_alive(p)]
    assert not alive, f"abort 后不应有存活进程: {alive}"


@check("05 HDR 两段式：音频 + hvc1 + colr 元数据齐全")
def c05():
    from sharp3d.hdr import Hdr10Writer
    out = TMP / "hdr_two_stage.mp4"
    out.unlink(missing_ok=True)
    w = Hdr10Writer(out, width=320, height=240, fps=12, codec="h265", crf=30,
                    audio_source=AUDIO)
    assert w._mux_proc is not None, "HDR 默认也应启用两段式"
    for i in range(40):
        w.write_frame(np.full((240, 320, 3), (i * 9) % 256, dtype=np.uint8))
    w.close()
    assert "audio" in streams(out), "HDR 成品应含音轨"
    assert not is_fragmented(out), "HDR 成品应非分片"
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries",
                        "stream=codec_tag_string,color_transfer,color_primaries",
                        "-of", "default=noprint_wrappers=1", str(out)],
                       capture_output=True)
    info = r.stdout.decode(errors="replace")
    assert "hvc1" in info, f"HEVC 需 hvc1 tag: {info}"
    assert "smpte2084" in info, f"HDR10 需 PQ: {info}"


def main():
    gen_audio()
    print(f"--- 实时音轨两段式验证 ({TMP}) ---")
    for name, fn in CHECKS:
        t0 = time.time()
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"PASS  {name}  ({time.time() - t0:.0f}s)")
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, repr(e)))
            print(f"FAIL  {name}: {e!r}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nRESULT {passed}/{len(RESULTS)} PASS")
    for n, ok, err in RESULTS:
        if not ok:
            print(f"  FAILED: {n} -> {err}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
