"""Lightweight per-stage profiling, enabled via SHARP3D_PROFILE=1.

Zero overhead when disabled (single module-level bool check per site).
CUDA stages are timed with cuda events (collected and drained in batches to
avoid per-frame synchronization); the ORT encoder is timed with wall clock
(run_with_iobinding blocks the CPU, so perf_counter around it is accurate).
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("SHARP3D_PROFILE") in ("1", "2", "3")
# =3 additionally enables CPU-side per-section wall timestamps in the worker
# loop (worker.py), to locate GPU-idle glue between the CUDA-event stages.
CPU_TRACE = os.environ.get("SHARP3D_PROFILE") == "3"

# Wall-clock ms accumulated inside ORTEncoder.forward, keyed by label.
ort_ms: dict[str, float] = {"patch_encoder": 0.0, "image_encoder": 0.0}


def add_ort(label: str, ms: float) -> None:
    ort_ms[label] = ort_ms.get(label, 0.0) + ms


def take_ort() -> float:
    """Return total accumulated ORT ms and reset the counters."""
    total = sum(ort_ms.values())
    for k in ort_ms:
        ort_ms[k] = 0.0
    return total


def vram_line() -> str:
    """One-line VRAM snapshot: torch allocator state + driver free memory."""
    import torch
    alloc = torch.cuda.memory_allocated() / 2**20
    reserved = torch.cuda.memory_reserved() / 2**20
    free, total = torch.cuda.mem_get_info()
    return (f"alloc={alloc:.0f}MB reserved={reserved:.0f}MB "
            f"free={free / 2**20:.0f}/{total / 2**20:.0f}MB")


_timer = None


def get_timer() -> "StageTimer":
    """Process-wide singleton timer for the video conversion stages."""
    global _timer
    if _timer is None:
        _timer = StageTimer(["predict", "stabilize", "render+pack"])
    return _timer


class StageTimer:
    """Batched CUDA-event stage timer for the video loop.

    Usage:
        timer = StageTimer(["predict", "stabilize", "render"])
        for frame:
            timer.frame_start()
            ... predict ...
            timer.mark("predict")
            ... stabilize ...
            timer.mark("stabilize")
            ... render ...
            timer.mark("render")
            timer.frame_end(extra_wall_ms)   # prints every `interval` frames
    """

    def __init__(self, stages: list[str], interval: int = 30):
        import torch
        self._torch = torch
        self._stages = stages
        self._interval = interval
        self._frames: list[list] = []   # [start_ev, ev_stage0, ev_stage1, ...]
        self._wall: list[float] = []    # extra wall-clock ms per frame
        self._count = 0

    def frame_start(self) -> None:
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        self._frames.append([ev])

    def mark(self, _stage: str) -> None:
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        self._frames[-1].append(ev)

    def frame_end(self, extra_wall_ms: float = 0.0) -> None:
        self._wall.append(extra_wall_ms)
        self._count += 1
        # SHARP3D_PROFILE=2: per-frame breakdown. The aggregated averages hide
        # the shape of the cost — a 5s recompile on frame 1 averages into the
        # same number as a uniform per-frame regression, and they need
        # completely different fixes.
        if os.environ.get("SHARP3D_PROFILE") == "2":
            evs = self._frames[-1]
            self._torch.cuda.synchronize()
            parts = []
            for i in range(min(len(self._stages), len(evs) - 1)):
                parts.append(f"{self._stages[i]}="
                             f"{evs[i].elapsed_time(evs[i + 1]):7.1f}ms")
            import torch as _t
            print(f"[PROF-frame {self._count:4d}] " + "  ".join(parts)
                  + f"  pipe={extra_wall_ms:6.1f}ms"
                  + f"  torch_mem={_t.cuda.memory_allocated()/2**20:.0f}"
                  f"/{_t.cuda.memory_reserved()/2**20:.0f}MB", flush=True)
        if self._count % self._interval == 0:
            self.flush()

    def flush(self) -> None:
        if not self._frames:
            return
        self._torch.cuda.synchronize()
        n = len(self._frames)
        sums = [0.0] * len(self._stages)
        for evs in self._frames:
            for i in range(min(len(self._stages), len(evs) - 1)):
                sums[i] += evs[i].elapsed_time(evs[i + 1])
        wall = sum(self._wall) / max(len(self._wall), 1)
        ort = take_ort() / n
        parts = [f"{name}={sums[i] / n:7.1f}ms"
                 for i, name in enumerate(self._stages)]
        total = sum(sums) / n + wall
        print(f"[PROF] n={n:3d}  " + "  ".join(parts)
              + f"  d2h+write={wall:6.1f}ms"
              + f"  | ort_enc={ort:6.1f}ms  total={total:7.1f}ms"
              f"  ({1000.0 / max(total, 1e-6):.2f} fps)")
        self._frames.clear()
        self._wall.clear()
