"""Verify batch 3: GUI job ownership, FIFO attribution, batch skip semantics.

Runs fully offscreen with a stub engine (same signal surface as
EngineProcess, no child process). The EngineProcess._poll FIFO logic is
driven directly with a canned response queue.

Acceptance (from the fix plan):
  3.1  A tab's done/error must not reset another tab's in-flight state.
  3.2  A handler exception must not swallow the rest of the response queue.
  3.3  codec mapping by combo index; every index resolves to a valid codec.
  3.4  A failing batch file is skipped, not retried forever.
  3.5  anim export refuses stale-parameter frames; duplicate prepare blocked.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, r"C:/Users/yhm/.qoderworkcn/workspace/mrsw6dewe12d0mgy/sharp3d/src")

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal                      # noqa: E402
from PySide6.QtWidgets import QApplication                      # noqa: E402

app = QApplication.instance() or QApplication([])  # QWidget needs QApplication

from sharp3d.gui.theme import ThemeManager                       # noqa: E402
from sharp3d.gui.sbs_tab import SbsTab, _CODECS as SBS_CODECS    # noqa: E402
from sharp3d.gui.vr_tab import VrTab, _CODECS as VR_CODECS       # noqa: E402
from sharp3d.gui.anim_tab import AnimTab, _CODECS as ANIM_CODECS # noqa: E402
from sharp3d.gui import worker as worker_mod                     # noqa: E402

fails: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {label}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(label)


class _CancelEvent:
    def __init__(self):
        self._flag = False

    def set(self):
        self._flag = True

    def clear(self):
        self._flag = False

    def is_set(self):
        return self._flag


class _FakeQueue:
    """Minimal stand-in for mp.Queue with canned messages."""

    def __init__(self, messages):
        self._msgs = list(messages)

    def get_nowait(self):
        if not self._msgs:
            raise worker_mod.queue.Empty()
        return self._msgs.pop(0)


class StubEngine(QObject):
    """Same signal surface as EngineProcess, no child process."""
    model_loading = Signal()
    model_load_progress = Signal(str, int)
    model_ready = Signal()
    model_accel = Signal(list)
    prepared = Signal(dict)
    preview_ready = Signal(object)
    convert_progress = Signal(int, int, float, float, int)
    convert_done = Signal(dict)
    anim_progress = Signal(int, int, int)
    anim_done = Signal(dict)
    anim_exported = Signal(str, int)
    ply_loaded = Signal(dict)
    orbit_frame = Signal(object)
    error = Signal(str, int)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self._seq = 0
        self.convert_calls: list[tuple[dict, int]] = []

    def new_job(self) -> int:
        self._seq += 1
        return self._seq

    def cancel(self):
        pass

    def convert(self, opts, job_id=-1):
        self.convert_calls.append((opts, job_id))

    def prepare(self, path, frame_idx, perf_mode="quality", focal_35mm=None,
                job_id=-1):
        pass

    def render_anim(self, opts, job_id=-1):
        pass

    def export_anim(self, opts, job_id=-1):
        pass


print("=" * 74)
print("3.1/3.2 EngineProcess._poll FIFO attribution + exception isolation")
print("=" * 74)
ep = worker_mod.EngineProcess.__new__(worker_mod.EngineProcess)
QObject.__init__(ep)  # init the Qt side only — no child process spawned
ep._active_jobs = worker_mod.deque()
ep._job_seq = 0
ep._cancel_event = _CancelEvent()
ep._cancel_event.set()  # simulate a pending cancel; idle clear must reset it

received: list[tuple] = []
ep.convert_done.connect(lambda payload: received.append(("done", payload["job_id"])))
ep.error.connect(lambda msg, job_id: received.append(("error", job_id)))
ep.convert_progress.connect(
    lambda f, t, fps, el, job: received.append(("prog", job)))

def boom(payload):
    raise RuntimeError("handler exploded")

ep.convert_done.connect(boom)  # 3.2: this must not kill the poll loop

ep._resp_q = _FakeQueue([
    # two jobs queued: A then B; a handler exception in between
    ("convert_done", ({"n_frames": 3, "fps": 1.0, "output": "a"},)),
    ("convert_done", ({"n_frames": 4, "fps": 1.0, "output": "b"},)),
    ("convert_progress", (1, 10, 2.0, 0.5)),
    ("error", ("late failure",)),
])
ep._active_jobs.extend([11, 12])
ep._job_seq = 12
ep._poll()

check(received[0] == ("done", 11), "first convert_done attributed to job A",
      f"{received[0]}")
check(received[1] == ("done", 12), "handler exception did not kill the loop; "
      "second done attributed to job B", f"{received[1]}")
check(received[2] == ("prog", -1), "progress after idle attributed to -1",
      f"{received[2]}")
check(received[3] == ("error", -1), "error after idle attributed to -1",
      f"{received[3]}")
check(not ep._cancel_event.is_set(),
      "cancel event cleared when the pipeline went idle")

print()
print("=" * 74)
print("3.1 tab isolation: A tab's signals leave B tab untouched")
print("=" * 74)
theme = ThemeManager()
eng = StubEngine()

sbs = SbsTab(theme, eng)
vr = VrTab(theme, eng)

# Both tabs "convert" concurrently: sbs owns job 1, vr owns job 2.
sbs._converting = True
sbs._job_id = 1
vr._converting = True
vr._job_id = 2

# Foreign signals first (while both are still in flight)…
eng.error.emit("boom", 1)
check(vr._converting and vr._job_id == 2,
      "sbs's error did not reset vr's in-flight state")
# …then each tab's own terminal signal.
eng.convert_done.emit({"n_frames": 3, "fps": 1.0, "output": "x",
                       "job_id": 1})
check(not sbs._converting and sbs._job_id == -1,
      "sbs's own convert_done resets sbs's state")
check(vr._converting,
      "sbs's own convert_done did not touch vr")
eng.error.emit("vr failed", 2)
check(not vr._converting and vr._btn_start.isEnabled(),
      "vr's own error resets vr's state and re-enables start")
eng.convert_done.emit({"n_frames": 3, "fps": 1.0, "output": "x",
                       "job_id": 2})
check(vr._btn_start.isEnabled(),
      "late convert_done after an error is ignored (job already closed)")

print()
print("=" * 74)
print("3.4 batch semantics: failing file is skipped, not retried")
print("=" * 74)
eng2 = StubEngine()
sbs2 = SbsTab(theme, eng2)
sbs2._batch_files = [r"C:\fake\a.jpg", r"C:\fake\b.jpg", r"C:\fake\c.jpg"]
sbs2._batch_idx = 0
sbs2._failed_files = []
sbs2._converting = True
sbs2._job_id = 1

# File 1 fails -> must advance to file 2 with a fresh job, stay converting.
eng2.error.emit("codec exploded", 1)
check(sbs2._batch_idx == 1 and sbs2._converting,
      "error advances to the next batch file",
      f"idx={sbs2._batch_idx}")
check(sbs2._failed_files == [r"C:\fake\a.jpg"],
      "failed file recorded", f"{sbs2._failed_files}")
check(len(eng2.convert_calls) == 1,
      "next file's convert request emitted", f"{len(eng2.convert_calls)}")

# File 2 fails -> advances to file 3.
eng2.error.emit("codec exploded again", sbs2._job_id)
check(sbs2._batch_idx == 2 and sbs2._converting and len(eng2.convert_calls) == 2,
      "second failure also skipped")

# File 3 fails -> batch finalises: state cleared, skipped count reported.
statuses: list[str] = []
sbs2.status_message.connect(statuses.append)
eng2.error.emit("boom", sbs2._job_id)
check(not sbs2._converting and not sbs2._batch_files,
      "last failure finalises the batch")
check(any("3 个文件" in s and "3 个失败" in s for s in statuses),
      "completion message reports the skipped count", f"{statuses[-1:]}")

print()
print("=" * 74)
print("3.3 codec maps resolve for every combo index")
print("=" * 74)
for name, tab, codecs, want in (
        ("sbs", sbs2, SBS_CODECS, ("av1", "h264", "h265")),
        ("vr", vr, VR_CODECS, ("av1", "h265", "h264"))):
    ok = codecs == want
    results = []
    for i in range(tab._codec.count()):
        tab._codec.setCurrentIndex(i)
        results.append(codecs[tab._codec.currentIndex()])
    check(ok and all(r in ("av1", "h264", "h265") for r in results),
          f"{name} tab codec map by index", f"{results}")

print()
print("=" * 74)
print("3.5 anim_tab guards")
print("=" * 74)
eng3 = StubEngine()
anim = AnimTab(theme, eng3)

# A foreign error must not reset an in-flight render (previously unguarded).
anim._rendering = True
anim._job_id = 5
anim._has_anim = True
eng3.error.emit("sbs tab blew up", 1)
check(anim._rendering, "foreign error leaves the render running")
eng3.error.emit("anim failed", 5)
check(not anim._rendering, "own error resets the render state")

# Export staleness check: frames rendered at (quality, None); the user then
# switches precision -> export must refuse (returns before the file dialog).
anim._has_anim = True
anim._rendered_params = ("quality", None)
anim._prec.setCurrentIndex(1)  # speed
statuses3: list[str] = []
anim.status_message.connect(statuses3.append)
anim._on_export()
check(any("旧参数" in s or "重新生成" in s for s in statuses3),
      "stale-parameter export is refused", f"{statuses3[-1:]}")

# Duplicate prepare guard: while awaiting a rebuild, a second render click
# must not emit another prepare request.
prep_calls: list[tuple] = []
anim.request_prepare.connect(lambda p, f, m, fo, j: prep_calls.append(j))
anim._prepared = True
anim._rendering = False
anim._awaiting_prepare = False
anim._input_path = r"C:\fake\img.jpg"   # rebuild branch requires an input
anim._prep_mode = "quality"
anim._prep_focal = None
anim._prec.setCurrentIndex(2)   # fp32 -> rebuild branch
anim._on_render()               # emits prepare, sets _awaiting_prepare
n_after_first = len(prep_calls)
anim._on_render()               # must be blocked by the preparing guard
check(n_after_first == 1 and len(prep_calls) == 1,
      "duplicate prepare blocked while a rebuild is in flight",
      f"prep_calls={len(prep_calls)}")

print()
print("=" * 74)
print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
for f in fails:
    print("   -", f)
sys.exit(0 if not fails else 1)
