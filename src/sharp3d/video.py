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
    if encoder == "av1_nvenc":
        # NVENC has no CRF; QP mode is the closest analogue.
        # ffmpeg 2026+ deprecated -cq (global_quality), use -qp instead.
        return ["-rc", "vbr", "-qp", str(min(crf, 51)),
                "-b:v", "0", "-preset", "p4"]
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        # Map preset names to NVENC p1-p7 scale (p4 ≈ medium balance).
        nv_preset = {"ultrafast": "p1", "fast": "p3", "medium": "p4",
                     "slow": "p6", "veryslow": "p7"}.get(preset, "p4")
        return ["-rc", "vbr", "-qp", str(min(crf, 51)),
                "-b:v", "0", "-preset", nv_preset]
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
    """Write video frames with encoding options (GPU-first encoder selection)."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int,
                 codec: str = "h264", crf: int = 18, preset: str = "medium"):
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
            FFMPEG, "-y",
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
