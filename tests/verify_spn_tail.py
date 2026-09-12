"""Verify SpnTail produces identical results to the torch full forward.

Pure-PyTorch A/B (no ONNX): run the full predictor once with ViT capture,
then feed the captured ViT outputs into SpnTail. fp32, same weights —
expect max diff ≤1e-4 on all 5 gaussian fields.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from export_full_model import build_predictor  # noqa: E402
from sharp3d.spn_tail import SpnTail, capture_vit_outputs  # noqa: E402
from sharp3d import resolve_cache_dir  # noqa: E402


def main():
    device = torch.device("cuda")
    predictor = build_predictor(device, "quality")

    from verify_full_onnx import grab_frame  # noqa: E402
    from sharp3d.unproject import prepare_input
    frame = grab_frame(ROOT.parent / "outputs" / "_bench_4k.mp4", 10)
    img_r, df_t, _ir, _sz = prepare_input(frame, 1920 * 1.2, device)

    g_ref, cap = capture_vit_outputs(predictor, img_r, df_t)

    tail = SpnTail(predictor, 35).eval()
    with torch.no_grad():
        outs = tail(img_r, df_t, cap["pe_feat"], cap["pe_int0"],
                    cap["pe_int1"], cap["ie_feat"])

    names = SpnTail._OUT_FIELDS
    ref = [getattr(g_ref, n) for n in names]
    ok = True
    for name, o, r in zip(names, outs, ref):
        d = (o.float() - r.float()).abs()
        good = d.max().item() < 1e-4
        ok &= good
        print(f"  {name:16s} max={d.max().item():.3e} mean={d.mean().item():.3e} "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print("TAIL_VERIFY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
