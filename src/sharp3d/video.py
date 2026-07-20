"""Video I/O: frame reading, writing, and audio muxing.

Uses imageio for frame I/O and ffmpeg subprocess for audio mux.
Encoding: H.264 (libx264 crf18) / H.265 (libx265) / AV1 (best available).
"""

import os
import subprocess
from pathlib import Path

import imageio
import numpy as np

from .hdr import FFMPEG

# imageio bundles its own minimal ffmpeg that may lack libsvtav1 (the AV1
# encoder the GUI offers). Pin it to the same full-featured binary the rest
# of the pipeline uses. Must run before the first imageio writer is created;
# a user-provided IMAGEIO_FFMPEG_EXE is respected.
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", FFMPEG)

# AV1 encoders in preference order; the first one present is used.
# GPU first: NVENC encodes ~10x faster than realtime on a dedicated engine
# (measured: no impact on render speed), software encoders as CPU fallback.
AV1_CHAIN = ("av1_nvenc", "libsvtav1", "libaom-av1")


# NVENC hardware encoding caps (pixels per side).
NVENC_LIMITS = {"h264_nvenc": 4096, "hevc_nvenc": 8192, "av1_nvenc": 8192}


def resolve_av1(width: int = 0, height: int = 0) -> str | None:
    """Pick the best available AV1 encoder for the given frame size.

    NVENC entries are skipped when the frame exceeds the hardware cap
    (e.g. 4320-wide 4K SBS -> 8640 wide is beyond the 8192 limit); software
    encoders have no such cap and serve as the fallback.
    """
    from .hdr import available_encoders

    avail = available_encoders()
    max_dim = max(width, height)
    for name in AV1_CHAIN:
        if name not in avail:
            continue
        if max_dim > NVENC_LIMITS.get(name, 1 << 30):
            continue
        return name
    return None


def av1_output_params(encoder: str, crf: int) -> list[str]:
    """ffmpeg output params for the given AV1 encoder at ~crf quality."""
    if encoder == "av1_nvenc":
        # NVENC has no CRF; CQ mode is the closest analogue. Default offset
        # +8 (CRF 18 -> CQ 26): NVENC is quality-cheap per bit, so the GPU
        # default runs a higher rate factor than the software encoders.
        return ["-rc", "vbr", "-cq", str(min(crf + 8, 51)),
                "-b:v", "0", "-preset", "p4"]
    if encoder == "libaom-av1":
        # libaom is very slow; raise encoding speed for near-realtime use.
        return ["-crf", str(crf), "-b:v", "0", "-cpu-used", "8", "-row-mt", "1"]
    return ["-crf", str(crf), "-preset", "6"]  # libsvtav1 (preset is 0-13)


class VideoReader:
    """Read video frames with metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.reader = imageio.get_reader(str(self.path))
        meta = self.reader.get_meta_data()
        self.width, self.height = meta["size"]
        self.fps = meta["fps"]
        self.n_frames = self.reader.count_frames()
        self.has_audio = meta.get("audio_codec") is not None

    def get_frame(self, idx: int) -> np.ndarray:
        """Get frame as (H, W, 3) uint8 numpy array."""
        return self.reader.get_data(idx)

    def close(self):
        self.reader.close()

    def __len__(self):
        return self.n_frames

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class VideoWriter:
    """Write video frames with encoding options."""

    CODEC_MAP = {
        "h264": ("libx264", ".mp4"),
        "h265": ("libx265", ".mp4"),
        "av1": ("libsvtav1", ".mp4"),
    }

    def __init__(self, path: str | Path, fps: float, width: int, height: int,
                 codec: str = "h264", crf: int = 18, preset: str = "medium"):
        self.path = Path(path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec_name = codec

        codec_lib, _ = self.CODEC_MAP.get(codec, ("libx264", ".mp4"))

        # Write to temp file first (audio mux later if needed)
        self.tmp_path = self.path.with_suffix(".tmp.mp4")

        if codec == "av1":
            # Not every ffmpeg build has libsvtav1, and NVENC cannot encode
            # beyond 8192 px per side — resolve against both availability and
            # the output size. Fail now (before rendering any frames) instead
            # of at the end of a long conversion.
            codec_lib = resolve_av1(width, height)
            if codec_lib is None:
                raise RuntimeError(
                    "当前 ffmpeg 不支持任何 AV1 编码器"
                    "（需要 libsvtav1 / av1_nvenc / libaom-av1 之一）"
                )
            output_params = av1_output_params(codec_lib, crf)
        else:
            output_params = ["-crf", str(crf), "-preset", preset]
            if codec == "h265":
                output_params.extend(["-tag:v", "hvc1"])

        self.writer = imageio.get_writer(
            str(self.tmp_path),
            fps=fps,
            codec=codec_lib,
            quality=8,
            pixelformat="yuv420p",
            output_params=output_params,
        )

    def append_frame(self, frame: np.ndarray):
        """Append (H, W, 3) uint8 frame."""
        self.writer.append_data(frame)

    def close(self, source_video: str | Path | None = None):
        """Close writer and optionally mux audio from source.

        Args:
            source_video: Path to original video for audio extraction.
        """
        self.writer.close()

        if source_video is not None:
            self._mux_audio(Path(source_video))
        else:
            self.tmp_path.replace(self.path)

    def _mux_audio(self, source: Path):
        """Mux audio from source video into output."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.tmp_path),
            "-i", str(source),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(self.path),
        ]
        # binary capture: only the return code matters; text decoding of
        # ffmpeg's stderr (which echoes CJK filenames as UTF-8) would crash
        # under the GBK locale.
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            self.tmp_path.unlink(missing_ok=True)
        else:
            # Fallback: keep video without audio
            self.tmp_path.replace(self.path)
