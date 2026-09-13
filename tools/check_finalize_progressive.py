"""验证 finished 输出为非分片 faststart MP4（VR/SMB 可打开、可快进）。

背景：转换中的 .tmp.mp4 是 fragmented MP4（边转边播用）；但 finished 文件
如果是分片的，就没有全局样本索引 —— VR 头显经 SMB 读取时无法打开/快进
（实测 moov 仅 1239B + 2077 个 moof，nb_frames=N/A）。close() 现在会把
成品重封装为传统 faststart MP4。

覆盖：
  1. finalize_progressive：分片 → 非分片，且解码数据逐帧一致（framemd5）
  2. 回退：remux 失败时直接改名，视频不丢（返回 False）
  3. SHARP3D_KEEP_FRAGMENTED=1：跳过封装，保持分片
  4. VideoWriter 无音频：close() 后成品非分片、帧数正确
  5. VideoWriter 有音频（legacy mux）：成品非分片 + 带 aac
  6. VideoWriter 实时音轨（opt-in）：成品非分片 + 带 aac
  7. Hdr10Writer：成品非分片 + hvc1 + smpte2084
"""
import importlib
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = Path(os.environ.get("TEMP", "/tmp")) / "sharp3d_finalize_test"
TMP.mkdir(parents=True, exist_ok=True)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

RESULTS = []
CHECKS = []


def check(name):
    """注册一个检查项（不在定义时执行——之前的写法会立刻跑一次并返回
    未包装的函数，main() 再跑一次时异常就逃逸了）。"""
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def boxes(path, want=()):
    """返回顶层 box 列表; want 中的类型附带 payload。"""
    size = os.path.getsize(path)
    out = {}
    seq = []
    pos = 0
    with open(path, "rb") as f:
        while pos + 8 <= size:
            f.seek(pos)
            hdr = f.read(16)
            if len(hdr) < 8:
                break
            bsize = struct.unpack(">I", hdr[:4])[0]
            btype = hdr[4:8].decode("latin1", "replace")
            hsize = 8
            if bsize == 1:
                bsize = struct.unpack(">Q", hdr[8:16])[0]
                hsize = 16
            elif bsize == 0:
                bsize = size - pos
            seq.append(btype)
            if btype in want:
                f.seek(pos)
                out[btype] = f.read(bsize)
            pos += max(bsize, hsize)
    return seq, out


def is_fragmented(path):
    seq, got = boxes(path, want=("moov",))
    moov = got.get("moov", b"")
    return (b"mvex" in moov), seq


def nb_frames(path, stream="v:0"):
    r = run([FFPROBE, "-v", "error", "-select_streams", stream,
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", str(path)])
    return r.stdout.decode().strip()


def framemd5(path, stream="v:0"):
    r = run([FFMPEG, "-v", "error", "-i", str(path), "-map", f"0:{stream}",
             "-f", "framemd5", "-"])
    return r.stdout.decode(errors="replace")


def make_source(dst, seconds=1, fps=12, size="320x240", audio=False):
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(dst)]
    r = run(cmd)
    assert r.returncode == 0, r.stderr.decode(errors="replace")[-500:]
    return dst


def make_fragmented(src, dst):
    """把普通 mp4 转成 fragmented mp4（模拟转换中的 .tmp 输出）。"""
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(src), "-c", "copy",
           "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
           "-frag_duration", "1000000", "-flush_packets", "1", str(dst)]
    r = run(cmd)
    assert r.returncode == 0, r.stderr.decode(errors="replace")[-500:]
    return dst


# ---------------------------------------------------------------- checks

@check("00 测试素材生成")
def c00():
    make_source(TMP / "plain.mp4")
    make_source(TMP / "plain_audio.mp4", audio=True)
    make_fragmented(TMP / "plain.mp4", TMP / "frag.mp4")
    frag, _ = is_fragmented(TMP / "frag.mp4")
    assert frag, "素材本身应是分片 MP4"
    plain, _ = is_fragmented(TMP / "plain.mp4")
    assert not plain


@check("01 finalize_progressive 产出非分片且数据逐帧一致")
def c01():
    from sharp3d.hdr import finalize_progressive
    src = TMP / "frag.mp4"
    src_copy = TMP / "frag_copy.mp4"
    src_copy.write_bytes(src.read_bytes())
    dst = TMP / "prog_out.mp4"
    dst.unlink(missing_ok=True)
    ok = finalize_progressive(src_copy, dst)
    assert ok is True, "封装应成功"
    frag, seq = is_fragmented(dst)
    assert not frag, f"成品不应是分片 (mvex found), boxes={seq[:8]}"
    assert "moof" not in seq, f"不应有 moof, boxes={seq[:8]}"
    _, got = boxes(dst, want=("moov",))
    moov = got["moov"]
    assert b"stsz" in moov and (b"co64" in moov or b"stco" in moov), \
        "moov 必须有完整样本表"
    assert not src_copy.exists(), "成功路径应删除临时源文件"
    # 数据等价（stream copy 应逐帧逐位一致）
    a, b = framemd5(TMP / "frag.mp4"), framemd5(dst)
    assert a and a == b, "重封装的解码数据必须与源逐帧一致"
    n = nb_frames(dst, "v:0")
    assert n == "12", f"帧数应为 12，实际 {n}"


@check("02 回退：remux 失败时改名保底，视频不丢")
def c02():
    from sharp3d.hdr import finalize_progressive
    bad = TMP / "not_a_video.mp4"
    bad.write_bytes(b"this is not a video at all")
    dst = TMP / "fallback_out.mp4"
    dst.unlink(missing_ok=True)
    ok = finalize_progressive(bad, dst)
    assert ok is False, "非视频输入应回退"
    assert dst.exists() and dst.read_bytes() == b"this is not a video at all", \
        "回退后成品应是原文件（内容不丢）"
    assert not bad.exists(), "回退路径应完成改名"


@check("03 KEEP_FRAGMENTED=1 时跳过封装（子进程隔离）")
def c03():
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from pathlib import Path;"
        "from sharp3d.hdr import finalize_progressive, KEEP_FRAGMENTED;"
        "assert KEEP_FRAGMENTED, 'KEEP_FRAGMENTED 应为 True';"
        "src = Path(r'%s'); dst = Path(r'%s');"
        "dst.unlink(missing_ok=True);"
        "ok = finalize_progressive(src, dst);"
        "print('OK' if ok is False else 'BAD')"
    ) % (ROOT / "src", TMP / "frag.mp4", TMP / "keep_out.mp4")
    # 复制一份避免消耗素材
    (TMP / "keep_src.mp4").write_bytes((TMP / "frag.mp4").read_bytes())
    code = code.replace(str(TMP / "frag.mp4"), str(TMP / "keep_src.mp4"))
    env = dict(os.environ, SHARP3D_KEEP_FRAGMENTED="1")
    env["PYTHONIOENCODING"] = "utf-8"
    r = run([sys.executable, "-c", code], env=env)
    out = r.stdout.decode(errors="replace")
    assert "OK" in out, f"应跳过封装并返回 False: {out} {r.stderr.decode(errors='replace')[-300:]}"
    frag, _ = is_fragmented(TMP / "keep_out.mp4")
    assert frag, "跳过时应保持分片"


@check("04 VideoWriter 无音频：成品非分片、帧数正确")
def c04():
    import numpy as np
    from sharp3d.video import VideoWriter
    out = TMP / "vw_plain.mp4"
    out.unlink(missing_ok=True)
    w = VideoWriter(out, fps=12, width=320, height=240, codec="h264", crf=30)
    for i in range(12):
        w.append_frame(np.full((240, 320, 3), (i * 20) % 256, dtype=np.uint8))
    w.close()
    assert out.exists(), "close() 后应产出成品"
    frag, seq = is_fragmented(out)
    assert not frag, f"成品不应分片: {seq[:8]}"
    assert nb_frames(out) == "12", f"帧数应 12，实际 {nb_frames(out)}"


@check("05 VideoWriter 有音频（legacy mux）：非分片 + aac")
def c05():
    import numpy as np
    from sharp3d.video import VideoWriter
    out = TMP / "vw_audio.mp4"
    out.unlink(missing_ok=True)
    w = VideoWriter(out, fps=12, width=320, height=240, codec="h264", crf=30,
                    audio_source=TMP / "plain_audio.mp4")
    for i in range(12):
        w.append_frame(np.full((240, 320, 3), (i * 20) % 256, dtype=np.uint8))
    w.close()
    frag, seq = is_fragmented(out)
    assert not frag, f"成品不应分片: {seq[:8]}"
    r = run([FFPROBE, "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(out)])
    kinds = r.stdout.decode().strip().split()
    assert "audio" in kinds, f"应含音轨, 实际 {kinds}"


@check("06 VideoWriter 实时音轨（SHARP3D_LIVE_AUDIO=1）：非分片 + aac")
def c06():
    code = (
        "import sys, numpy as np; sys.path.insert(0, r'%s');"
        "from pathlib import Path; from sharp3d.video import VideoWriter;"
        "out = Path(r'%s'); out.unlink(missing_ok=True);"
        "w = VideoWriter(out, fps=12, width=320, height=240, codec='h264', crf=30,"
        " audio_source=Path(r'%s'));"
        "[w.append_frame(np.full((240,320,3), (i*20)%%256, dtype=np.uint8)) for i in range(12)];"
        "w.close(); print('DONE')"
    ) % (ROOT / "src", TMP / "vw_live.mp4", TMP / "plain_audio.mp4")
    env = dict(os.environ, SHARP3D_LIVE_AUDIO="1", PYTHONIOENCODING="utf-8")
    r = run([sys.executable, "-c", code], env=env)
    assert "DONE" in r.stdout.decode(errors="replace"), \
        f"实时音轨转换应完成: {r.stdout.decode(errors='replace')[-300:]} " \
        f"{r.stderr.decode(errors='replace')[-400:]}"
    out = TMP / "vw_live.mp4"
    frag, seq = is_fragmented(out)
    assert not frag, f"成品不应分片: {seq[:8]}"
    rr = run([FFPROBE, "-v", "error", "-show_entries", "stream=codec_type",
              "-of", "csv=p=0", str(out)])
    assert "audio" in rr.stdout.decode().strip().split()


@check("07 Hdr10Writer：非分片 + hvc1 + smpte2084")
def c07():
    import numpy as np
    from sharp3d.hdr import Hdr10Writer
    out = TMP / "hdr_out.mp4"
    out.unlink(missing_ok=True)
    w = Hdr10Writer(out, width=320, height=240, fps=12, codec="h265", crf=30)
    for i in range(12):
        w.write_frame(np.full((240, 320, 3), (i * 20) % 256, dtype=np.uint8))
    w.close()
    frag, seq = is_fragmented(out)
    assert not frag, f"HDR 成品不应分片: {seq[:8]}"
    r = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_tag_string,color_transfer,color_primaries,pix_fmt",
             "-of", "default=noprint_wrappers=1", str(out)])
    info = r.stdout.decode()
    assert "hvc1" in info, f"HEVC 应为 hvc1 tag: {info}"
    # 字段名是 color_transfer（旧版 ffprobe 的 color_trc 别名在本版本返回空，
    # 实测 colr box 里的 transfer=16/smpte2084 一直是对的）。
    assert "smpte2084" in info, f"HDR10 需 PQ 传输特性: {info}"


def main():
    print(f"--- sharp3d finalize 验证 ({TMP}) ---")
    for name, fn in CHECKS:
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, repr(e)))
            print(f"FAIL  {name}: {e!r}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nRESULT {passed}/{len(RESULTS)} PASS")
    for name, ok, err in RESULTS:
        if not ok:
            print(f"  FAILED: {name} -> {err}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
