"""HDR handling for sharp3d.

The SHARP model is fundamentally SDR (sRGB input, linearRGB output, values in
[0,1]). True HDR output (real high dynamic range / wide gamut) is therefore
impossible — the 3D reconstruction happens entirely in SDR space.

What this module provides is *HDR10 format* input/output:
  - INPUT:  HDR sources are detected and properly tone-mapped to SDR before
            being fed to the model (so geometry/color prediction is correct,
            unlike a naive 8-bit decode which mangles PQ/BT.2020 content).
  - OUTPUT: the SDR render is converted to a valid HDR10 stream (10-bit,
            PQ/SMPTE-2084 transfer, BT.2020 gamut) so it displays at correct
            brightness on HDR devices without banding or wash-out.

All conversions use ffmpeg's zscale (zimg) filters, verified working.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 隐藏 Windows 子进程控制台窗口
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _exe(name: str) -> str:
    """Locate an ffmpeg/ffprobe executable."""
    found = shutil.which(name)
    if found:
        return found
    # common install location fallback
    candidate = Path(r"C:\Program Files\ffmpeg\bin") / f"{name}.exe"
    if candidate.exists():
        return str(candidate)
    return name  # hope it's on PATH


FFMPEG = _exe("ffmpeg")
FFPROBE = _exe("ffprobe")

# Transfer functions that indicate HDR content.
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10) and HLG

# Fragmented MP4 with per-packet flushing and a 1s fragment cap, so the
# output stays playable while it is still being written. Both extra flags
# are required — see sharp3d.video.MOVFLAGS_LIVE for the measurements.
# Duplicated rather than imported: video.py imports this module for FFMPEG,
# so importing back would create a cycle.
MOVFLAGS_LIVE = ["-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                 "-frag_duration", "1000000",
                 "-flush_packets", "1"]

# Fragmented output is what makes mid-conversion playback possible, but it
# is a poor *finished* file: the moov carries no sample table, so a player
# must walk every moof to seek — or give up. Measured on the same 8K SBS
# production output (2026-09-13):
#   fragmented    : moov 1,239 B (mvex, nb_frames=N/A) + 2,077 moof/mdat
#                   → VR headset over SMB could not open or seek it
#   non-fragmented: moov 2.1 MB with a full co64/stsz sample table,
#                   nb_frames=57,867 → played and seeked fine
# So every finished output is remuxed into a plain faststart MP4 (`-c copy`,
# no re-encode — only the container is rewritten). Set
# SHARP3D_KEEP_FRAGMENTED=1 to skip that pass: cheaper (no 2x file I/O) but
# the result stays fragmented and may not play/seek on hardware players.
KEEP_FRAGMENTED = os.environ.get("SHARP3D_KEEP_FRAGMENTED") == "1"


def hvc1_tag_args(codec_lib: str) -> list[str]:
    """`-tag:v hvc1` for HEVC, nothing otherwise.

    Accepts both the family name ("h265") and the resolved encoder
    ("hevc_nvenc"/"libx265"); hardware players (and Apple's stack) expect
    the hvc1 sample entry for HEVC in MP4.
    """
    return (["-tag:v", "hvc1"]
            if ("hevc" in codec_lib or "265" in codec_lib) else [])


def finalize_progressive(src: Path, dst: Path,
                         extra: list[str] | None = None) -> bool:
    """Promote a finished fragmented MP4 (`src`) to a compatible MP4 at `dst`.

    Remuxes to a non-fragmented, faststart MP4 so ordinary players can open
    and seek the file. On any failure it falls back to a plain rename, so
    the encoded video is never lost — it just stays fragmented.

    `extra` carries encoder flags that must survive the remux (currently the
    hvc1 tag for HEVC, which several hardware players require).

    Returns True when the progressive remux succeeded, False when the
    fragmented fallback was used. Raises OSError only if the fallback rename
    also fails (destination locked or on another device).
    """
    if not KEEP_FRAGMENTED:
        cmd = [FFMPEG, "-y", "-i", str(src), "-c", "copy",
               "-map", "0:v:0", "-map", "0:a:0?",
               "-movflags", "+faststart"]
        if extra:
            cmd += extra
        cmd.append(str(dst))
        logger.info("正在封装为兼容 MP4（传统样本索引 + moov 前置）: %s", dst.name)
        try:
            result = subprocess.run(cmd, capture_output=True,
                                    creationflags=_NO_WINDOW)
        except OSError as e:
            result = None
            logger.error("调用 ffmpeg 封装兼容 MP4 失败: %s", e)
        if result is not None and result.returncode == 0:
            try:
                src.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("兼容封装成功但临时文件删除失败: %s", e)
            return True
        tail = ""
        if result is not None and result.stderr:
            tail = result.stderr.decode("utf-8",
                                        errors="replace").strip()[-500:]
        logger.error("封装为兼容 MP4 失败（返回码 %s）；已保留分片输出，"
                     "部分播放器可能无法打开或快进，可设 "
                     "SHARP3D_KEEP_FRAGMENTED=0 后重试本次转换: %s\n%s",
                     "n/a" if result is None else result.returncode, dst, tail)
        try:
            dst.unlink(missing_ok=True)  # 清掉半成品，避免误认
        except OSError:
            logger.debug("兼容封装半成品删除失败: %s", dst, exc_info=True)

    try:
        src.replace(dst)
    except OSError as e:
        logger.error("无法将临时文件移为最终输出（%s → %s）: %s；"
                     "视频仍保留在 %s", src, dst, e, src)
        raise
    if KEEP_FRAGMENTED:
        logger.info("SHARP3D_KEEP_FRAGMENTED=1：保留分片输出（未封装为兼容 MP4）")
    return False


# --- Encoder capability probing ----------------------------------------------

_ENCODERS: set[str] | None = None
# GUI worker subprocesses and pipeline codec threads can both hit this on
# first use; the lock keeps the probe single-flight.
_ENCODERS_LOCK = threading.Lock()


def available_encoders() -> set[str]:
    """Video-encoder names supported by the ffmpeg in use (cached).

    Queries IMAGEIO_FFMPEG_EXE when set (imageio writers honor it), else the
    FFMPEG binary resolved above — in practice both are the same executable.

    A *failed* probe (ffmpeg missing, spawn error) is deliberately NOT
    cached: the old code stored the empty set, which made
    ``encoder_available()`` false forever and silently downgraded every
    later encode to the CPU chain with no way to recover. Only a successful
    probe (even an empty-looking output) is cached; failures retry on the
    next call.
    """
    global _ENCODERS
    with _ENCODERS_LOCK:
        if _ENCODERS is None:
            import os

            exe = os.environ.get("IMAGEIO_FFMPEG_EXE") or FFMPEG
            try:
                out = subprocess.run(
                    [exe, "-hide_banner", "-encoders"], capture_output=True,
                    creationflags=_NO_WINDOW,
                ).stdout.decode("utf-8", "replace")
                names: set[str] = set()
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].startswith("V"):
                        names.add(parts[1])
                _ENCODERS = names
            except Exception as e:
                logger.warning("ffmpeg 编码器探测失败（不缓存，下次调用重试）: %s", e)
        return _ENCODERS if _ENCODERS is not None else set()


def encoder_available(name: str) -> bool:
    return name in available_encoders()


# --- Hardware decode (NVDEC) probing -----------------------------------------

_HWACCEL_CUDA: bool | None = None


def hwaccel_cuda_available() -> bool:
    """Check if ffmpeg supports CUDA hardware decoding (NVDEC). Cached."""
    global _HWACCEL_CUDA
    if _HWACCEL_CUDA is None:
        try:
            out = subprocess.run(
                [FFMPEG, "-hide_banner", "-hwaccels"],
                capture_output=True,
                creationflags=_NO_WINDOW,
            ).stdout.decode("utf-8", "replace")
            _HWACCEL_CUDA = "cuda" in out.lower()
        except Exception:
            _HWACCEL_CUDA = False
    return _HWACCEL_CUDA


# --- Filter chains ----------------------------------------------------------

def hdr_to_sdr_filter(peak: float = 1000.0) -> str:
    """HDR (PQ/HLG, BT.2020) -> SDR (BT.709 gamma) tone-mapping filter.

    Linearizes the EOTF, maps the wide gamut to BT.709, applies a Hable
    tone-mapping curve, then re-encodes to BT.709 gamma. Output is rgb24.
    """
    return (
        "zscale=t=linear:npl=100,format=gbrpf32le,"
        "zscale=p=bt709,"
        f"tonemap=hable:desat=0:peak={peak},"
        "zscale=t=bt709:m=bt709:r=tv,format=rgb24"
    )


def sdr_to_hdr10_filter() -> str:
    """SDR (BT.709 gamma, rgb24) -> HDR10 (10-bit PQ, BT.2020) filter.

    Declares the input colorspace explicitly (required by zscale, otherwise
    it fails with 'no path between colorspaces'), converts gamut to BT.2020
    and applies the PQ transfer. Output is yuv420p10le.
    """
    return (
        "format=yuv420p10le,"
        "zscale=matrixin=bt709:primariesin=bt709:transferin=bt709:rangein=tv:"
        "matrix=bt2020nc:primaries=bt2020:transfer=smpte2084:range=tv"
    )


# Standard BT.2020 1000-nit mastering display metadata for x265 (HDR10).
MASTER_DISPLAY = "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1)"
MAX_CLL = "1000,400"
# NVENC wants the same values but '|'-separated groups.
MASTER_DISPLAY_NVENC = ("G(8500,39850)|B(6550,2300)|R(35400,14600)|"
                        "WP(15635,16450)|L(10000000,1)")


_NVENC_HDR10: bool | None = None


def nvenc_hdr10_capable() -> bool:
    """True if hevc_nvenc supports HDR10 metadata (-master_display). Cached."""
    global _NVENC_HDR10
    if _NVENC_HDR10 is None:
        ok = False
        if encoder_available("hevc_nvenc"):
            try:
                out = subprocess.run(
                    [FFMPEG, "-hide_banner", "-h", "encoder=hevc_nvenc"],
                    capture_output=True, creationflags=_NO_WINDOW,
                ).stdout.decode("utf-8", "replace")
                ok = "master_display" in out
            except Exception:
                ok = False
        _NVENC_HDR10 = ok
    return _NVENC_HDR10


# --- Probing ----------------------------------------------------------------

def probe_video(path: str | Path) -> dict:
    """Probe a video file and return metadata including HDR detection.

    Returns a dict with: width, height, fps, n_frames, has_audio, is_hdr,
    color_transfer, color_primaries, pix_fmt, bit_depth.

    Raises ValueError if the file has no decodable video stream or reports a
    non-positive size/frame count. Returning zeros instead used to be silent
    and lethal downstream: pipeline computes the output size and f_px from
    these numbers, so a 0x0 probe produced a ``-s 0x0`` ffmpeg invocation
    (broken file) or a zero focal length, with no error ever surfaced.
    """
    cmd = [
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    # ffprobe emits UTF-8 (the JSON embeds the filename); text=True alone
    # would decode as GBK on Chinese Windows and crash on CJK filenames.
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace",
                            creationflags=_NO_WINDOW)
    data = json.loads(result.stdout) if result.stdout else {}

    vstream = None
    has_audio = False
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and vstream is None:
            vstream = s
        elif s.get("codec_type") == "audio":
            has_audio = True

    info = {
        "width": 0, "height": 0, "fps": 30.0, "n_frames": 0,
        "has_audio": has_audio, "is_hdr": False,
        "color_transfer": None, "color_primaries": None,
        "pix_fmt": None, "bit_depth": 8,
    }
    if vstream is None:
        if has_audio:
            raise ValueError(
                f"该文件只有音频流，没有视频流: {path}")
        raise ValueError(
            f"无法读取视频流信息（文件损坏或格式不支持）: {path}")

    info["width"] = int(vstream.get("width", 0) or 0)
    info["height"] = int(vstream.get("height", 0) or 0)
    info["pix_fmt"] = vstream.get("pix_fmt")
    info["color_transfer"] = vstream.get("color_transfer")
    info["color_primaries"] = vstream.get("color_primaries")

    if info["width"] <= 0 or info["height"] <= 0:
        raise ValueError(
            f"视频尺寸无效（{info['width']}x{info['height']}）: {path}")

    # fps from avg_frame_rate "num/den"
    afr = vstream.get("avg_frame_rate", "30/1")
    try:
        num, den = afr.split("/")
        info["fps"] = float(num) / float(den) if float(den) else 30.0
    except (ValueError, ZeroDivisionError):
        info["fps"] = 30.0
    # Clamp: r_frame_rate/avg_frame_rate are frequently "0/0" or absurd for
    # VFR and screen-capture sources, and this value divides into per-frame
    # timestamps and the ETA elsewhere.
    if not (1.0 <= info["fps"] <= 1000.0):
        logger.warning("异常的 fps=%s (avg_frame_rate=%r)，回退 30.0: %s",
                       info["fps"], afr, path)
        info["fps"] = 30.0

    # frame count (nb_frames may be N/A; fall back to duration * fps)
    nb = vstream.get("nb_frames")
    if nb and nb != "N/A":
        info["n_frames"] = int(nb)
    else:
        dur = float(data.get("format", {}).get("duration", 0) or 0)
        info["n_frames"] = int(round(dur * info["fps"]))

    if info["n_frames"] <= 0:
        # Without a frame count the conversion loop never runs yet reports
        # success with nan fps (np.mean of an empty list) — reject up front.
        raise ValueError(
            f"无法确定帧数（duration/nb_frames 缺失或为 0）: {path}")

    # bit depth from pix_fmt (e.g. yuv420p10le -> 10)
    pf = info["pix_fmt"] or ""
    if "16le" in pf or "16be" in pf:
        info["bit_depth"] = 16
    elif "12le" in pf or "12be" in pf:
        info["bit_depth"] = 12
    elif "10le" in pf or "10be" in pf:
        info["bit_depth"] = 10

    # HDR detection: PQ/HLG transfer, or 10-bit+ with BT.2020 primaries
    ct = (info["color_transfer"] or "").lower()
    cp = (info["color_primaries"] or "").lower()
    info["is_hdr"] = (ct in HDR_TRANSFERS) or (
        info["bit_depth"] >= 10 and cp == "bt2020"
    )

    return info


# --- Frame reading (with HDR->SDR tone-mapping) -----------------------------

class FrameReader:
    """Reads video frames as SDR rgb24 numpy arrays.

    If the source is HDR, frames are tone-mapped to SDR during decode so the
    SHARP model receives correct input.
    Uses NVDEC hardware decoding when available (offloads CPU).
    """

    def __init__(self, path: str | Path, info: dict | None = None):
        self.path = str(path)
        self.info = info or probe_video(path)
        self.width = self.info["width"]
        self.height = self.info["height"]
        self.fps = self.info["fps"]
        self.n_frames = self.info["n_frames"]
        self.is_hdr = self.info["is_hdr"]
        self.has_audio = self.info["has_audio"]
        self._frame_size = self.width * self.height * 3

    def _hwaccel(self) -> list[str]:
        """NVDEC hardware decode flags (GPU engine, zero CPU cost)."""
        if hwaccel_cuda_available():
            return ["-hwaccel", "cuda"]
        return []

    def _vf(self) -> list[str]:
        if self.is_hdr:
            return ["-vf", hdr_to_sdr_filter()]
        return []

    def _vf_select(self, idx: int) -> list[str]:
        """Filter chain that emits exactly frame ``idx``, tone-mapped if HDR.

        Selects by frame number rather than seeking by timestamp: ``idx / fps``
        is wrong for VFR sources, and input-side ``-ss`` (fast seek) decodes
        from the nearest *keyframe* at or after the target, so the frame
        returned was not the one requested — a silent preview mismatch.
        """
        select = f"select=eq(n\\,{idx})"
        if self.is_hdr:
            return ["-vf", f"{select},{hdr_to_sdr_filter()}"]
        return ["-vf", select]

    def read_frame(self, idx: int) -> np.ndarray:
        """Read a single frame by index. For previews."""
        cmd = [FFMPEG, *self._hwaccel(), "-i", self.path,
               *self._vf_select(idx), "-vframes", "1",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        result = subprocess.run(cmd, capture_output=True,
                                creationflags=_NO_WINDOW)
        raw = result.stdout
        if len(raw) < self._frame_size:
            raise RuntimeError(f"Failed to decode frame {idx}")
        return np.frombuffer(raw[: self._frame_size], dtype=np.uint8).reshape(
            self.height, self.width, 3
        ).copy()

    def stream_frames(self):
        """Generator yielding all frames in order. For full conversion."""
        cmd = [FFMPEG, *self._hwaccel(),
               "-i", self.path, *self._vf(),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                creationflags=_NO_WINDOW)
        try:
            while True:
                raw = proc.stdout.read(self._frame_size)
                if len(raw) < self._frame_size:
                    break
                yield np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                ).copy()
        finally:
            self._reap(proc)

    @staticmethod
    def _reap(proc: subprocess.Popen) -> None:
        """Terminate an ffmpeg child without leaking it or raising.

        Order matters: close the pipe first so a writer blocked on a full pipe
        gets EPIPE, then wait briefly, and only kill as a last resort. Killing
        unconditionally raised PermissionError on Windows for already-exited
        processes (swallowed by the old bare finally), and the caller's
        ``join(timeout=5)`` could leave the thread blocked in
        ``proc.stdout.read`` with the child still alive.
        """
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    def close(self) -> None:
        """No persistent handle today; kept so callers can bracket usage."""
        return None


# --- HDR10 writing ----------------------------------------------------------

class Hdr10Writer:
    """Encodes SDR rgb24 frames into an HDR10 (10-bit PQ BT.2020) video.

    Frames are piped raw into ffmpeg, which applies the SDR->HDR10 conversion
    and encodes with libx265 (or libsvtav1) including HDR10 metadata.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float,
                 codec: str = "h265", crf: int = 18,
                 audio_source: str | Path | None = None):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        # Live-audio mode（两段式）：音频随视频写进同一个 fragmented 容器，
        # 转换中就能听到（原理与实测见 VideoWriter 的对应注释）。
        # SHARP3D_LIVE_AUDIO=0 可显式关闭，退回"转换完成后复用"。
        self._live_audio = audio_source is not None
        self._close_mux_source = None
        if self._live_audio and os.environ.get("SHARP3D_LIVE_AUDIO") == "0":
            logger.info("实时音轨已按 SHARP3D_LIVE_AUDIO=0 关闭；"
                        "音频将在转换完成后复用")
            # 保存源路径，close() 时走完成后复用路径
            self._close_mux_source = Path(audio_source)
            self._live_audio = False
        # Clamp both ends: a negative CRF/qp is rejected by every encoder
        # here, and 0 (near-lossless constqp) can explode the file size.
        crf = max(0, min(int(crf), 51))
        # write to a temp file, mux audio later
        self.tmp_path = self.path.with_suffix(".tmp.mp4")

        # Prefer hevc_nvenc for HEVC HDR10: the hardware encoder is ~10x
        # faster than libx265 and carries the HDR10 static metadata itself
        # (ffmpeg ≥ 6; capability is probed once above).
        if (codec == "h265" and nvenc_hdr10_capable()
                and max(width, height) <= 8192):
            v_codec = "hevc_nvenc"
            enc_params = [
                "-profile:v", "main10",
                "-rc", "vbr", "-qp", str(min(crf, 51)), "-b:v", "0",
                "-preset", "p4",
                "-master_display", MASTER_DISPLAY_NVENC,
                "-max_cll", MAX_CLL,
            ]
        elif codec == "av1" and encoder_available("libsvtav1"):
            # SVT-AV1 can carry the HDR10 static metadata itself.
            v_codec = "libsvtav1"
            enc_params = ["-svtav1-params",
                          f"crf={crf}:master-display={MASTER_DISPLAY}:"
                          f"max-cll={MAX_CLL}"]
        else:
            # No SVT-AV1 (e.g. a stripped ffmpeg build): other AV1 encoders
            # can't inject the HDR10 metadata reliably, so fall back to the
            # proven libx265 HDR10 path. (h264 can't do HDR10 at all.)
            v_codec = "libx265"
            xparams = (
                f"crf={crf}:hdr10=1:repeat-headers=1:"
                f"master-display={MASTER_DISPLAY}:max-cll={MAX_CLL}"
            )
            enc_params = ["-x265-params", xparams]

        self._v_codec = v_codec
        self._mux_proc = None

        if self._live_audio:
            # 两段式（原理与实测见 VideoWriter）：同一进程里做视频编码 +
            # 音频编码/mux 会让 ffmpeg 稳定占住约 7GB。中间容器 AV1 用 ivf、
            # HEVC 用 mpegts —— AV1 不进 mpegts 的原因见 VideoWriter
            # （写入成功但读回只认成 bin_data，闭环走不通）。
            es_fmt = "ivf" if "av1" in v_codec else "mpegts"
            enc_cmd = [
                FFMPEG, "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-vf", sdr_to_hdr10_filter(),
                "-c:v", v_codec, *enc_params,
                "-pix_fmt", "yuv420p10le",
                # 逐包刷出，否则编码进程的 avio 缓冲会把中间流攒住，
                # 混流进程收不到内容（边转边播失效，详见 VideoWriter）
                "-flush_packets", "1",
                "-f", es_fmt, "-",
            ]
            mux_cmd = [
                FFMPEG, "-y",
                # 限定输入探测窗口（默认 5MB 会让 ffmpeg#2 等到管道 EOF 才
                # 建流，转换中完全没有输出；给太大也不行——低码率内容填不满。
                # 详见 VideoWriter 的说明）
                "-probesize", "4096", "-analyzeduration", "0",
                "-f", es_fmt, "-i", "-",
                "-i", str(Path(audio_source)),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                # HDR 色彩标记必须挂在容器侧：中间流（mpegts/ivf）不承载
                # colr 信息，只在编码侧给会让重封装后的 colr box 丢失
                # （实测 transfer 变成 unknown）。
                "-color_primaries", "bt2020",
                "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc",
                "-color_range", "tv",
                *hvc1_tag_args(v_codec),
                *MOVFLAGS_LIVE,
                str(self.tmp_path),
            ]
            self._mux_proc = subprocess.Popen(
                mux_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW)
            self._proc = subprocess.Popen(
                enc_cmd, stdin=subprocess.PIPE, stdout=self._mux_proc.stdin,
                stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
            # 写端交给编码进程（OS 直连），Python 不保留副本
            self._mux_proc.stdin.close()
        else:
            # 无实时音轨：单进程编码，音频留给 close() 复用
            cmd = [
                FFMPEG, "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-vf", sdr_to_hdr10_filter(),
                "-c:v", v_codec, *enc_params,
                "-pix_fmt", "yuv420p10le",
                "-color_primaries", "bt2020",
                "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc",
                "-color_range", "tv",
                *MOVFLAGS_LIVE, str(self.tmp_path),
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL,
                                          creationflags=_NO_WINDOW)

    def write_frame(self, frame: np.ndarray) -> None:
        """Write an (H, W, 3) uint8 SDR frame."""
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        # memoryview hands the buffer straight to the pipe writer; the old
        # frame.tobytes() made a full-frame copy (~100MB at 8K) every frame.
        self._proc.stdin.write(memoryview(frame))

    def close(self, audio_source: str | Path | None = None) -> None:
        """Finish encoding and optionally mux audio from a source video."""
        # 门控关闭的实时音轨：构造时传入的 audio_source 落到 close 时复用
        if audio_source is None and self._close_mux_source is not None:
            audio_source = self._close_mux_source
        self._proc.stdin.close()
        rc_enc = self._proc.wait()
        # 两段式：编码进程退出后，混流进程才收到管道 EOF
        rc_mux = self._mux_proc.wait() if self._mux_proc is not None else 0
        if rc_enc != 0 or rc_mux != 0:
            raise RuntimeError(
                f"HDR10 编码器异常退出 (enc={rc_enc}, mux={rc_mux})")

        if audio_source is not None and not self._live_audio:
            # The conversion is finished, so compatibility beats
            # live-playback: emit a plain faststart MP4 instead of
            # re-fragmenting. A fragmented result has no global sample
            # index — a VR headset reading one over SMB could neither open
            # nor seek it (see finalize_progressive for the measurements).
            cmd = [
                FFMPEG, "-y",
                "-i", str(self.tmp_path),
                "-i", str(audio_source),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                "-movflags", "+faststart",
            ]
            cmd += hvc1_tag_args(self._v_codec)
            cmd.append(str(self.path))
            result = subprocess.run(cmd, capture_output=True,
                                    creationflags=_NO_WINDOW)
            if result.returncode == 0:
                try:
                    self.tmp_path.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning("音频复用成功但临时文件删除失败: %s", e)
                return
            # Losing the audio must not be silent (see VideoWriter._mux_audio).
            tail = ""
            if result.stderr:
                tail = result.stderr.decode("utf-8",
                                            errors="replace").strip()[-500:]
            logger.error("HDR10 音频复用失败（返回码 %d），已保留无音频视频: %s\n%s",
                         result.returncode, self.path, tail)

        finalize_progressive(self.tmp_path, self.path,
                             hvc1_tag_args(self._v_codec))

    def abort(self) -> None:
        """Best-effort cleanup after a failed/cancelled run. Never raises.

        Closes the encoder's stdin so ffmpeg cannot block on a full pipe,
        stops the process, and removes the tmp file: without this an
        exception between writer creation and close() leaked a live ffmpeg
        child plus a .tmp.mp4 that looked like a valid output.
        """
        procs = [self._proc]
        if getattr(self, "_mux_proc", None) is not None:
            procs.append(self._mux_proc)
        # 两段式下要收掉两个进程：先关写端，再依次 terminate/kill
        for p in procs:
            try:
                p.stdin.close()
            except Exception:
                pass
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        try:
            self.tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("abort: 临时文件删除失败: %s", self.tmp_path,
                         exc_info=True)
