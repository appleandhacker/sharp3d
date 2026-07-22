"""Stereo output format packing.

Every format is derived from the rendered Full SBS image (left|right) by
slicing, bilinear resampling and channel compositing on the GPU — no extra
scene rendering is needed for any format.

Formats (mirroring the common iw3/nunif viewing formats):
  full_sbs  : left | right, full resolution per eye      (2W × H)
  half_sbs  : left | right, each eye squeezed to W/2     (W × H)
  full_tb   : left over right, full resolution           (W × 2H)
  half_tb   : left over right, each eye squeezed to H/2  (W × H)
  cross     : right | left (cross-eyed viewing)          (2W × H)
  anaglyph  : red/cyan half-color composite              (W × H)
"""

from __future__ import annotations

# NOTE: torch is imported lazily inside pack()/_squeeze() so that the GUI
# process can import the FORMATS metadata without loading torch.

# (key, display label) — order = GUI dropdown order.
FORMATS: list[tuple[str, str]] = [
    ("full_sbs", "Full SBS"),
    ("half_sbs", "Half SBS"),
    ("full_tb", "Full TB"),
    ("half_tb", "Half TB"),
    ("cross", "Cross Eyed"),
    ("anaglyph", "Anaglyph 红青"),
]

FORMAT_KEYS = [k for k, _ in FORMATS]
FORMAT_LABELS = {k: label for k, label in FORMATS}


def _even(x: int) -> int:
    """Round down to even (video codecs require even dimensions)."""
    return x - (x % 2)


def output_size(fmt: str, w: int, h: int) -> tuple[int, int]:
    """Output (width, height) for one eye of w×h packed as `fmt`."""
    if fmt in ("full_sbs", "cross"):
        return _even(w * 2), _even(h)
    if fmt == "full_tb":
        return _even(w), _even(h * 2)
    # half_sbs / half_tb / anaglyph
    return _even(w), _even(h)


def _squeeze(img: "torch.Tensor", size: tuple[int, int]) -> "torch.Tensor":
    """(H, W, 3) uint8 -> bilinear-resized (size_h, size_w, 3) uint8."""
    import torch
    import torch.nn.functional as F

    t = img.permute(2, 0, 1)[None].float()
    out = F.interpolate(t, size=size, mode="bilinear", align_corners=True)
    return out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)


def pack(fmt: str, sbs: "torch.Tensor") -> "torch.Tensor":
    """Pack a Full SBS render into the requested output format.

    Args:
        fmt: one of FORMAT_KEYS.
        sbs: (H, 2W, 3) uint8 tensor (left | right).

    Returns:
        Packed uint8 tensor; see output_size() for its shape.
    """
    import torch

    h, w2, _ = sbs.shape
    w = w2 // 2
    left, right = sbs[:, :w], sbs[:, w:]

    if fmt == "full_sbs":
        out = sbs
    elif fmt == "cross":
        out = torch.cat([right, left], dim=1)
    elif fmt == "full_tb":
        out = torch.cat([left, right], dim=0)
    elif fmt == "anaglyph":
        out = torch.stack([left[..., 0], right[..., 1], right[..., 2]], dim=-1)
    elif fmt == "half_sbs":
        half_w = _even(w // 2)
        out = torch.cat([_squeeze(left, (h, half_w)),
                         _squeeze(right, (h, half_w))], dim=1)
    elif fmt == "half_tb":
        half_h = _even(h // 2)
        out = torch.cat([_squeeze(left, (half_h, w)),
                         _squeeze(right, (half_h, w))], dim=0)
    else:
        raise ValueError(f"unknown stereo format: {fmt}")

    # Enforce even dimensions (video codecs require this).
    oh, ow = out.shape[0], out.shape[1]
    return out[:_even(oh), :_even(ow)]
