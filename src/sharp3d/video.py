"""Video I/O: frame reading, writing, and audio muxing.

Uses imageio for frame I/O and ffmpeg subprocess for audio mux.
Encoding: H.264 (libx264 crf18) / H.265 (libx265) / AV1 (libsvtav1).
"""

import subprocess
from pathlib import Path

import imageio
import numpy as np


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
