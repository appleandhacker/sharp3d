"""Video I/O: frame reading, writing, and audio muxing.

Uses imageio for frame I/O and ffmpeg subprocess for audio mux.
Encoding: H.264 (libx264 crf18) / H.265 (libx265) / AV1 (best available).
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import imageio
import numpy as np

from .hdr import FFMPEG, finalize_progressive, hvc1_tag_args

logger = logging.getLogger(__name__)

# 隐藏 Windows 子进程控制台窗口
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_IMAGEIO_FFMPEG_PINNED = False


def _pin_imageio_ffmpeg() -> None:
    """Pin imageio to the same full-featured ffmpeg binary the pipeline uses.

    imageio bundles its own minimal ffmpeg that may lack libsvtav1 (the AV1
    encoder the GUI offers). Must run before the first imageio reader/writer
    is created; a user-provided IMAGEIO_FFMPEG_EXE is respected.

    Deferred to first use instead of import time: resolving FFMPEG at import
    and writing it into the environment froze a possibly-bare "ffmpeg" (when
    nothing was on PATH) into IMAGEIO_FFMPEG_EXE, so imageio failed later with
    an error pointing at the wrong binary. Note importing this module already
    imports .hdr, which resolved FFMPEG the same way — deferring only helps
    if PATH changes between import and first writer creation, and keeps the
    environment untouched for importers that never do video I/O.
    """
    global _IMAGEIO_FFMPEG_PINNED
    if not _IMAGEIO_FFMPEG_PINNED:
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", FFMPEG)
        _IMAGEIO_FFMPEG_PINNED = True

# Encoder preference: GPU (NVENC) first, CPU fallback.
# NVENC runs on a dedicated hardware engine — ~10x faster than software,
# zero impact on CUDA render performance.
ENCODER_CHAINS = {
    "h264": ("h264_nvenc", "libx264"),
    "h265": ("hevc_nvenc", "libx265"),
    "av1": ("av1_nvenc", "libsvtav1", "libaom-av1"),
}

# NVENC hardware encoding caps (pixels per side).
NVENC_LIMITS = {"h264_nvenc": 4096, "hevc_nvenc": 8192, "av1_nvenc": 8192}

# File extension per codec family.
CODEC_EXT = {"h264": ".mp4", "h265": ".mp4", "av1": ".mp4"}

# Fragmented MP4, so the file becomes readable as fragments are flushed
# instead of only after the muxer writes its single trailing moov index at
# close(). Lets a player open the output mid-conversion and seek through
# whatever has been encoded so far.
#
# Two non-obvious requirements, both measured on ffmpeg 7.1 (640x368@30,
# frames piped through stdin):
#   -flush_packets 1      ffmpeg buffers output packets in userspace by
#                         default, so the movflags alone leave the file at a
#                         bare 28-byte ftyp header until close(). Without
#                         this flag nothing is gained at all.
#   -frag_duration 1s     fragment boundaries are otherwise driven purely by
#                         keyframes, and none of the encoders here set a
#                         short GOP: with the x264 default (250 frames) the
#                         first readable fragment appeared only at frame 250.
#                         A 1s cap makes that ~30 frames regardless of the
#                         encoder's GOP.
# Verified: first readable at frame 270 before this change, frame 50 after,
# with libx264 and h264_nvenc behaving identically.
MOVFLAGS_LIVE = ["-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                 "-frag_duration", "1000000",
                 "-flush_packets", "1"]


def resolve_encoder(codec: str, width: int = 0, height: int = 0) -> str:
    """Pick the best available encoder: GPU first, CPU fallback.

    NVENC entries are skipped when the frame exceeds the hardware cap
    (e.g. 4K SBS -> 8640 wide is beyond the 8192 limit for HEVC/AV1).
    """
    from .hdr import available_encoders

    avail = available_encoders()
    max_dim = max(width, height)
    chain = ENCODER_CHAINS.get(codec, ("libx264",))
    for name in chain:
        if name not in avail:
            continue
        if max_dim > NVENC_LIMITS.get(name, 1 << 30):
            continue
        return name
    # Last resort: return the CPU encoder even if probing failed
    return chain[-1]


# Keep legacy alias for external callers (worker.py uses resolve_av1).
def resolve_av1(width: int = 0, height: int = 0) -> str | None:
    """Pick the best available AV1 encoder for the given frame size."""
    from .hdr import available_encoders

    avail = available_encoders()
    max_dim = max(width, height)
    for name in ENCODER_CHAINS["av1"]:
        if name not in avail:
            continue
        if max_dim > NVENC_LIMITS.get(name, 1 << 30):
            continue
        return name
    return None


def _is_nvenc(encoder: str) -> bool:
    return "nvenc" in encoder


def encoder_output_params(encoder: str, crf: int, preset: str = "medium") -> list[str]:
    """ffmpeg output params for the given encoder at ~crf quality."""
    # Clamp both ends: every encoder here rejects negative CRF/qp, and 0
    # (near-lossless) can explode the file size.
    crf = max(0, min(int(crf), 51))
    if encoder == "av1_nvenc":
        # NVENC has no CRF; constqp is the closest analogue. Measured
        # (2026-09-11): -qp under -rc vbr is silently IGNORED — constqp
        # actually honors it.
        # SCALE WARNING (measured 2026-09-13): av1_nvenc's -qp runs on the
        # AV1 native quantizer scale 0-255, NOT the H.264-style 0-51 scale
        # the CRF slider uses (qp 40 ≈ 12.8 Mbps @1080p; the old clamp to 51
        # made low-bitrate targets unreachable). Map CRF 0-51 → qindex 0-255
        # linearly (crf 18→90, 26→130, 40→200, 51→255) and clamp the MAPPED
        # value, not the input.
        crf = max(0, min(int(crf), 51)) * 5
        return ["-rc", "constqp", "-qp", str(min(crf, 255)),
                "-preset", "p4"]
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        # Map preset names to NVENC p1-p7 scale (p4 ≈ medium balance).
        nv_preset = {"ultrafast": "p1", "fast": "p3", "medium": "p4",
                     "slow": "p6", "veryslow": "p7"}.get(preset, "p4")
        # constqp, NOT vbr+qp: same measurement as above — vbr drops -qp
        # and falls back to default-rate control, making every CRF value
        # produce identical bitrate.
        return ["-rc", "constqp", "-qp", str(min(crf, 51)),
                "-preset", nv_preset]
    if encoder == "libaom-av1":
        # libaom is very slow; raise encoding speed for near-realtime use.
        return ["-crf", str(crf), "-b:v", "0", "-cpu-used", "8", "-row-mt", "1"]
    if encoder == "libsvtav1":
        return ["-crf", str(crf), "-preset", "6"]  # preset is 0-13
    # libx264 / libx265
    return ["-crf", str(crf), "-preset", preset]


# Legacy alias
def av1_output_params(encoder: str, crf: int) -> list[str]:
    return encoder_output_params(encoder, crf)


class VideoReader:
    """Read video frames with metadata."""

    def __init__(self, path: str | Path):
        _pin_imageio_ffmpeg()
        self.path = Path(path)
        self.reader = None
        try:
            self.reader = imageio.get_reader(str(self.path))
            meta = self.reader.get_meta_data()
            # Some containers / damaged files come back without size or fps;
            # indexing raised KeyError (and leaked the open reader). Fail
            # loudly on an unusable size instead of producing a 0-size writer.
            width, height = (meta.get("size") or (0, 0))
            self.width, self.height = int(width), int(height)
            self.fps = float(meta.get("fps") or 30.0)
            if self.width <= 0 or self.height <= 0:
                raise ValueError(
                    f"视频尺寸无效（{self.width}x{self.height}）: {self.path}")
            self.n_frames = self.reader.count_frames()
            self.has_audio = meta.get("audio_codec") is not None
        except Exception:
            self.close()
            raise

    def get_frame(self, idx: int) -> np.ndarray:
        """Get frame as (H, W, 3) uint8 numpy array."""
        return self.reader.get_data(idx)

    def close(self):
        if getattr(self, "reader", None) is not None:
            try:
                self.reader.close()
            except Exception:
                logger.debug("VideoReader.close() 失败（忽略）", exc_info=True)
            self.reader = None

    def __len__(self):
        return self.n_frames

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class VideoWriter:
    """Write video frames with encoding options (GPU-first encoder selection)."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int,
                 codec: str = "h264", crf: int = 18, preset: str = "medium",
                 audio_source: str | Path | None = None):
        _pin_imageio_ffmpeg()
        self.path = Path(path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec_name = codec

        # GPU-first encoder resolution (NVENC → CPU fallback)
        codec_lib = resolve_encoder(codec, width, height)
        ext = CODEC_EXT.get(codec, ".mp4")

        # Write to temp file first (audio mux later if needed)
        self.tmp_path = self.path.with_suffix(".tmp" + ext)

        output_params = encoder_output_params(codec_lib, crf, preset)
        if codec == "h265":
            output_params.extend(["-tag:v", "hvc1"])
        # Fragmented container: append last so it applies to every encoder
        # branch, and lands before the output path imageio appends.
        output_params.extend(MOVFLAGS_LIVE)

        self._proc = None
        self._stderr_fh = None
        self._stderr_path = None
        self._close_mux_source = None
        self._audio_source = Path(audio_source) if audio_source is not None else None
        if (self._audio_source is not None
                and os.environ.get("SHARP3D_LIVE_AUDIO") != "1"):
            # 已知问题（2026-09-13 实测，8K AV1 NVENC / ffmpeg 7.1）：双输入
            # （视频管道 + 音频文件）下编码器内存随视频帧无界增长——30s 窗口
            # 2.95GB→10.1GB（+1.2GB/2s ≈ 每输出帧滞留一帧 23.5MB 的视频帧
            # 缓冲），32 分钟转换实测达 ~9-10GB；无音轨对照组平稳 2.95GB。
            # 增长与音频数据量无关（5MB 的 wav 同样触发），-re 限速未能确认
            # 有效。默认回退为"完成后复用"；确要实时音轨可设
            # SHARP3D_LIVE_AUDIO=1 强制启用（自行承担内存占用）。
            logger.info("实时音轨已禁用（ffmpeg 双输入下编码器内存无界增长）；"
                        "音频将在转换完成后复用。可设 SHARP3D_LIVE_AUDIO=1 "
                        "强制启用")
            # 保存源路径，close() 时走完成后复用路径（调用方仍是无参 close()）
            self._close_mux_source = self._audio_source
            self._audio_source = None

        if self._audio_source is not None:
            # Live-audio mode: mux the audio track into the SAME fragmented
            # output while video frames are written, so mid-conversion
            # playback has sound (the old close-time mux meant the tmp file
            # stayed silent until the whole conversion finished). imageio's
            # writer cannot take a second input, so run the ffmpeg pipe
            # directly — same pattern as Hdr10Writer. When the frame size is
            # not a multiple of 16, replicate imageio's macro_block_size
            # scale so output dimensions stay identical to the legacy path.
            self._stderr_path = self.tmp_path.with_suffix(".stderr")
            self._stderr_fh = open(self._stderr_path, "wb")
            cmd = [
                FFMPEG, "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-i", str(self._audio_source),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-pix_fmt", "yuv420p",
            ]
            if width % 16 or height % 16:
                cmd += ["-vf", f"scale={width + (16 - width % 16) % 16}:"
                               f"{height + (16 - height % 16) % 16}"]
            cmd += ["-c:v", codec_lib, *output_params,
                    "-c:a", "aac", "-shortest",
                    str(self.tmp_path)]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=self._stderr_fh,
                creationflags=_NO_WINDOW)
            logger.info("实时音轨已启用（音频随视频写入，转换中可听）: %s",
                        self._audio_source)
        else:
            # nvenc uses -qp in output_params; passing quality= would add the
            # deprecated -global_quality flag and trigger ffmpeg warnings/errors.
            writer_kwargs = dict(
                fps=fps,
                codec=codec_lib,
                pixelformat="yuv420p",
                output_params=output_params,
            )
            if not _is_nvenc(codec_lib):
                writer_kwargs["quality"] = 8

            self.writer = imageio.get_writer(str(self.tmp_path), **writer_kwargs)

    def append_frame(self, frame: np.ndarray):
        """Append (H, W, 3) uint8 frame."""
        if self._proc is not None:
            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)
            # memoryview hands the buffer to the pipe without a full-frame
            # tobytes() copy.
            self._proc.stdin.write(memoryview(frame))
        else:
            self.writer.append_data(frame)

    def close(self, source_video: str | Path | None = None):
        """Close writer and optionally mux audio from source.

        Args:
            source_video: Path to original video for audio extraction.
                Ignored in live-audio mode (the audio track was muxed into
                the stream as frames were written).
        """
        # 门控关闭的实时音轨：构造时传入的 audio_source 落到 close 时复用
        if source_video is None and getattr(self, "_close_mux_source", None):
            source_video = self._close_mux_source

        if self._proc is not None:
            # Live-audio mode: the audio track is already in the fragmented
            # stream; just finish the encode and promote tmp → final.
            self._proc.stdin.close()
            rc = self._proc.wait()
            self._close_stderr()
            if rc != 0:
                logger.error("实时音频编码器异常退出 (code=%d): %s\n%s",
                             rc, self.path, self._stderr_tail())
                raise RuntimeError(f"视频编码器异常退出 (code={rc})")
            self._unlink_stderr()
            if source_video is not None:
                logger.warning("已启用转换中音频，close(source_video=...) 被忽略")
            self._finalize()
            return

        self.writer.close()

        if source_video is not None:
            self._mux_audio(Path(source_video))
        else:
            self._finalize()

    def _close_stderr(self) -> None:
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except Exception:
                pass
            self._stderr_fh = None

    def _stderr_tail(self) -> str:
        try:
            return self._stderr_path.read_bytes()[-500:].decode(
                "utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _unlink_stderr(self) -> None:
        try:
            self._stderr_path.unlink(missing_ok=True)
        except (OSError, AttributeError):
            pass

    def _finalize(self) -> None:
        """Promote tmp → final path as a finished, compatible MP4.

        The encoder wrote a fragmented MP4 (live-playable mid-conversion,
        see MOVFLAGS_LIVE); finalize_progressive remuxes it into a
        faststart MP4 so ordinary players can open and seek it, and falls
        back to a plain rename if that fails — the video is never lost.
        """
        finalize_progressive(self.tmp_path, self.path,
                             hvc1_tag_args(self.codec_name))

    def abort(self) -> None:
        """Best-effort cleanup after a failed/cancelled run. Never raises.

        Closes the encoder (shutting down its ffmpeg child — the imageio
        writer in legacy mode, the direct pipe in live-audio mode) and
        removes the tmp file, so an exception between writer creation and
        close() neither leaks the encoder subprocess nor leaves a partial
        .tmp.mp4 that could be mistaken for a finished output.
        """
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._close_stderr()
        else:
            try:
                self.writer.close()
            except Exception:
                logger.debug("abort: writer.close() 失败（忽略）", exc_info=True)
        try:
            self.tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("abort: 临时文件删除失败: %s", self.tmp_path,
                         exc_info=True)
        self._unlink_stderr()

    def _mux_audio(self, source: Path):
        """Mux audio from source video into output."""
        cmd = [
            FFMPEG, "-y",
            "-i", str(self.tmp_path),
            "-i", str(source),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            # The conversion is finished, so compatibility beats
            # live-playback: emit a plain faststart MP4 instead of
            # re-fragmenting. A fragmented result carries no global sample
            # index — a VR headset reading one over SMB could neither open
            # nor seek it (see hdr.finalize_progressive for the measurements).
            "-movflags", "+faststart",
        ]
        cmd += hvc1_tag_args(self.codec_name)
        cmd.append(str(self.path))
        # binary capture: only the return code matters; text decoding of
        # ffmpeg's stderr (which echoes CJK filenames as UTF-8) would crash
        # under the GBK locale.
        result = subprocess.run(cmd, capture_output=True,
                                creationflags=_NO_WINDOW)
        if result.returncode == 0:
            try:
                self.tmp_path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("音频复用成功但临时文件删除失败: %s", e)
            return

        # Audio mux failed. Falling back to the silent video is the right call
        # (the video is complete and valuable), but it must be *visible*: this
        # previously happened with no message at all, so users received
        # audio-less output believing the conversion succeeded.
        tail = ""
        if result.stderr:
            tail = result.stderr.decode("utf-8", errors="replace").strip()[-500:]
        logger.error("音频复用失败（返回码 %d），已保留无音频视频: %s\n%s",
                     result.returncode, self.path, tail)
        self._finalize()
