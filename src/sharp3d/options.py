"""Typed conversion options (replaces untyped dict passing)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConvertOptions:
    """All parameters for a single SBS conversion job.

    Provides type safety, IDE autocomplete, and a single source of truth
    for defaults. Replaces the legacy untyped dict passed between GUI and worker.
    """

    input: str = ""
    output: str = ""

    # Stereo parameters
    format: str = "full_sbs"
    ipd_mm: float = 63.0
    convergence: float = 0.0       # 0 = auto
    strength: float = 1.0

    # Encoding
    codec: str = "h264"            # "h264" | "h265" | "av1"
    crf: int = 26
    audio: bool = True
    hdr_output: bool = False

    # Output geometry
    out_fps: float | None = None   # None = follow source
    out_scale: float = 1.0
    out_width: int | None = None   # None = use out_scale

    # Processing
    decompose: str = "analytical"  # "analytical" | "svd"
    perf_mode: str = "quality"     # "quality" | "speed"
    temporal_stabilize: str = "off"  # "off" | "global" | "adaptive" | "flow"

    # Extra outputs
    depth: bool = False
    ply: bool = False

    def to_dict(self) -> dict:
        """Convert to legacy dict for backward compatibility."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConvertOptions":
        """Create from legacy dict (ignores unknown keys)."""
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
