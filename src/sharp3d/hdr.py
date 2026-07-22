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
import shutil
import subprocess
from pathlib import Path

import numpy as np


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


# --- Encoder capability probing ----------------------------------------------

_ENCODERS: set[str] | None = None


def available_encoders() -> set[str]:
    """Video-encoder names supported by the ffmpeg in use (cached).

    Queries IMAGEIO_FFMPEG_EXE when set (imageio writers honor it), else the
    FFMPEG binary resolved above — in practice both are the same executable.
    """
    global _ENCODERS
    if _ENCODERS is None:
        import os

        exe = os.environ.get("IMAGEIO_FFMPEG_EXE") or FFMPEG
        names: set[str] = set()
        try:
            out = subprocess.run(
                [exe, "-hide_banner", "-encoders"], capture_output=True
            ).stdout.decode("utf-8", "replace")
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].startswith("V"):
                    names.add(parts[1])
        except Exception:
            pass
        _ENCODERS = names
    return _ENCODERS


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


# --- Probing ----------------------------------------------------------------

def probe_video(path: str | Path) -> dict:
    """Probe a video file and return metadata including HDR detection.

    Returns a dict with: width, height, fps, n_frames, has_audio, is_hdr,
    color_transfer, color_primaries, pix_fmt, bit_depth.
    """
    cmd = [
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    # ffprobe emits UTF-8 (the JSON embeds the filename); text=True alone
    # would decode as GBK on Chinese Windows and crash on CJK filenames.
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
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
        return info

    info["width"] = int(vstream.get("width", 0))
    info["height"] = int(vstream.get("height", 0))
    info["pix_fmt"] = vstream.get("pix_fmt")
    info["color_transfer"] = vstream.get("color_transfer")
    info["color_primaries"] = vstream.get("color_primaries")

    # fps from avg_frame_rate "num/den"
    afr = vstream.get("avg_frame_rate", "30/1")
    try:
        num, den = afr.split("/")
        info["fps"] = float(num) / float(den) if float(den) else 30.0
    except (ValueError, ZeroDivisionError):
        info["fps"] = 30.0

    # frame count (nb_frames may be N/A; fall back to duration * fps)
    nb = vstream.get("nb_frames")
    if nb and nb != "N/A":
        info["n_frames"] = int(nb)
    else:
        dur = float(data.get("format", {}).get("duration", 0) or 0)
        info["n_frames"] = int(round(dur * info["fps"]))

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

    def read_frame(self, idx: int) -> np.ndarray:
        """Read a single frame by index (time-based seek). For previews."""
        t = idx / self.fps if self.fps else 0.0
        cmd = [FFMPEG, *self._hwaccel(), "-ss", f"{t:.4f}", "-i", self.path,
               "-vframes", "1", *self._vf(),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        result = subprocess.run(cmd, capture_output=True)
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
                                stderr=subprocess.DEVNULL)
        try:
            while True:
                raw = proc.stdout.read(self._frame_size)
                if len(raw) < self._frame_size:
                    break
                yield np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                ).copy()
        finally:
            proc.stdout.close()
            proc.wait()


# --- HDR10 writing ----------------------------------------------------------

class Hdr10Writer:
    """Encodes SDR rgb24 frames into an HDR10 (10-bit PQ BT.2020) video.

    Frames are piped raw into ffmpeg, which applies the SDR->HDR10 conversion
    and encodes with libx265 (or libsvtav1) including HDR10 metadata.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float,
                 codec: str = "h265", crf: int = 18):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        # write to a temp file, mux audio later
        self.tmp_path = self.path.with_suffix(".tmp.mp4")

        if codec == "av1" and encoder_available("libsvtav1"):
            # SVT-AV1 can carry the HDR10 static metadata itself.
            v_codec = "libsvtav1"
            enc_params = ["-svtav1-params",
                          f"crf={crf}:master-display={MASTER_DISPLAY}:"
                          f"max-cll={MAX_CLL}"]
            color_opts = []
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
            color_opts = []

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
            str(self.tmp_path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL)

    def write_frame(self, frame: np.ndarray) -> None:
        """Write an (H, W, 3) uint8 SDR frame."""
        self._proc.stdin.write(frame.tobytes())

    def close(self, audio_source: str | Path | None = None) -> None:
        """Finish encoding and optionally mux audio from a source video."""
        self._proc.stdin.close()
        self._proc.wait()

        if audio_source is not None:
            cmd = [
                FFMPEG, "-y",
                "-i", str(self.tmp_path),
                "-i", str(audio_source),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                str(self.path),
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                self.tmp_path.unlink(missing_ok=True)
            else:
                self.tmp_path.replace(self.path)
        else:
            self.tmp_path.replace(self.path)
