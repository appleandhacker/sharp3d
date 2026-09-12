"""Verify task 1.6: audio-mux failure is logged, never silent.

Two scenarios must hold for VideoWriter:
  A) bogus audio source  -> mux fails -> ERROR logged, final video exists
     with audio stream count == 0, NO .tmp.mp4 residue.
  B) valid audio source  -> mux succeeds -> no error, final video HAS an
     audio stream, NO .tmp.mp4 residue.

Frame size is 256x256: h264_nvenc rejects frames below ~145x49, and the
earlier 64x64 attempt died inside append_frame before close() was reached,
so the mux path was never exercised.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:/Users/yhm/.qoderworkcn/workspace/mrsw6dewe12d0mgy/sharp3d/src")

from sharp3d.hdr import FFMPEG                       # noqa: E402
from sharp3d.video import VideoWriter                # noqa: E402

W = H = 256
NFRAMES = 15

records: list[tuple[int, str, str]] = []


class _Capture(logging.Handler):
    def emit(self, record):
        records.append((record.levelno, record.name, record.getMessage()))


logging.getLogger("sharp3d.video").addHandler(_Capture())
logging.getLogger("sharp3d.video").setLevel(logging.DEBUG)
logging.getLogger("sharp3d.video").propagate = False

TMP = Path(tempfile.gettempdir()) / "_t16_run"
TMP.mkdir(exist_ok=True)

for f in TMP.iterdir():
    f.unlink()

# A valid audio track to isolate "mux works" from "mux fails".
good_audio = TMP / "good.m4a"
subprocess.run(
    [FFMPEG, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
     "-c:a", "aac", "-b:a", "64k", str(good_audio)],
    capture_output=True, check=True,
)
bad_audio = TMP / "bad.mp4"
bad_audio.write_text("this is not a video at all", encoding="utf-8")


def audio_stream_count(path: Path) -> int:
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)],
        capture_output=True,
    )
    txt = r.stderr.decode("utf-8", errors="replace")
    return txt.count("Audio:")


def run_case(name: str, audio: Path) -> dict:
    records.clear()
    out = TMP / f"{name}.mp4"
    writer = VideoWriter(out, fps=15, width=W, height=H, codec="libx264")
    rng = np.random.default_rng(7)
    for _ in range(NFRAMES):
        writer.append_frame(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    writer.close(source_video=audio)

    tmp_left = list(TMP.glob(f"{name}.tmp*"))
    errs = [m for (_l, _n, m) in records if _l >= logging.ERROR]
    return {
        "name": name,
        "exists": out.exists(),
        "size": out.stat().st_size if out.exists() else 0,
        "tmp_left": [p.name for p in tmp_left],
        "errors": errs,
        "audio_streams": audio_stream_count(out) if out.exists() else -1,
    }


bad = run_case("bogus", bad_audio)
good = run_case("valid", good_audio)

WIDTH = 74
def line(tag, text):
    print(f"[{tag:4}] {text}")


print("=" * WIDTH)
print("A) bogus audio (mux must fail loudly)")
print("=" * WIDTH)
line("file", f"exists={bad['exists']} size={bad['size']}B")
line("tmp", f"residue={bad['tmp_left']}")
line("a#", f"audio streams = {bad['audio_streams']}")
for e in bad["errors"]:
    print(f"[ERR ] {e[:200]}")
ok_a = (bad["exists"] and bad["size"] > 0 and not bad["tmp_left"]
        and bad["audio_streams"] == 0 and len(bad["errors"]) >= 1
        and "音频复用失败" in bad["errors"][0])
line("A", "PASS" if ok_a else "FAIL")

print()
print("=" * WIDTH)
print("B) valid audio (mux must succeed silently)")
print("=" * WIDTH)
line("file", f"exists={good['exists']} size={good['size']}B")
line("tmp", f"residue={good['tmp_left']}")
line("a#", f"audio streams = {good['audio_streams']}")
for e in good["errors"]:
    print(f"[ERR ] {e[:200]}")
ok_b = (good["exists"] and good["size"] > 0 and not good["tmp_left"]
        and good["audio_streams"] == 1 and len(good["errors"]) == 0)
line("B", "PASS" if ok_b else "FAIL")

print()
print("RESULT:", "PASS" if (ok_a and ok_b) else "FAIL")
sys.exit(0 if (ok_a and ok_b) else 1)
