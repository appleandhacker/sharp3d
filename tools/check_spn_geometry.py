"""Verify the SPN split export honours n_patches, at the ONNX graph level.

This is the strongest GPU-free check available: it traces _SpnFront with the
real torch.onnx.export used by the pipeline (dynamo=False) for both 35 and 21
patches on CPU, then asserts the traced graph contains the right Split
grouping and that 21 does NOT contain the 35 grouping. It also runs the 21
dummy inputs through a CPU ONNX Runtime session to prove the graph is
executable — i.e. the merge padding matches the grid that split produced.

Needs no GPU, no SHARP weights, no TRT: it uses a tiny stand-in SPN built from
the real merge() so only the patch-grouping logic is under test.
"""
import os
import sys
from pathlib import Path

SHARP3D_SRC = os.environ.get("SHARP3D_SRC")
if not SHARP3D_SRC:
    raise SystemExit("set SHARP3D_SRC to the sharp3d/src directory")
sys.path.insert(0, SHARP3D_SRC)
sys.path.insert(0, str(Path(SHARP3D_SRC).parent.parent / "ml-sharp" / "src"))

import torch
import torch.nn as nn

from sharp.models.encoders.spn_encoder import merge
from sharp3d.full_export import _spn_geometry

OUT = Path(os.environ.get("TEMP", ".")) / "_sharp3d_spn_geom"
OUT.mkdir(parents=True, exist_ok=True)


# --- A stand-in SPN carrying just the merge/pad arithmetic -------------------
# Real convs would need the SHARP weights; the bug under test is purely which
# slice/split/padding values reach merge(), so identity-ish convs suffice.
class _FakeSPN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x1 convs keep channel counts arbitrary while staying traceable.
        self._c = nn.Conv2d(4, 4, 1)
        # fuse_lowres sees cat(x2, lowres) = 8 channels, as upstream's
        # fuse_lowres expects dims_encoder[4] * 2 in.
        self._fuse = nn.Conv2d(8, 4, 1)
        # No gradients needed: this harness never trains and the tracer
        # rejects requires_grad tensors as graph constants.
        for p in self.parameters():
            p.requires_grad_(False)

    def _up(self, x):
        return self._c(x)

    def upsample_latent0(self, x):
        return self._up(x)

    def upsample_latent1(self, x):
        return self._up(x)

    def upsample0(self, x):
        return self._up(x)

    def upsample1(self, x):
        return self._up(x)

    def upsample2(self, x):
        # Real upsample2/upsample_lowres each stride-2 an already-24x24 map to
        # 48x48; keep that so the fuse_lowres concat lines up.
        return nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    def upsample_lowres(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    def fuse_lowres(self, x):
        return self._fuse(x)


class _FakeViT(nn.Module):
    """reshape_feature: (B,577,C) -> (B,C,24,24) after dropping the CLS token."""

    def reshape_feature(self, e):
        b, _, c = e.shape
        return e[:, 1:, :].reshape(b, 24, 24, c).permute(0, 3, 1, 2)


class _FakePredictor(nn.Module):
    def __init__(self, spn):
        super().__init__()
        self.monodepth_model = nn.Module()
        md = nn.Module()
        md.encoder = spn
        self.monodepth_model.monodepth_predictor = md


def build_front(n_patches):
    """Mirror _SpnFront without importing it (it needs a real predictor)."""
    x0, split_sizes, padding = _spn_geometry(n_patches)
    spn = _FakeSPN()
    pred = _FakePredictor(spn)

    class Front(nn.Module):
        def forward(self, pe_feat, pe_int0, pe_int1, ie_feat):
            x_latent0 = spn.upsample_latent0(
                merge(spn.patch_encoder.reshape_feature(pe_int0)[:x0],
                      batch_size=1, padding=padding))
            x_latent1 = spn.upsample_latent1(
                merge(spn.patch_encoder.reshape_feature(pe_int1)[:x0],
                      batch_size=1, padding=padding))
            x0e, x1e, x2e = torch.split(pe_feat, split_sizes, dim=0)
            a = spn.upsample0(merge(x0e, batch_size=1, padding=padding))
            b = spn.upsample1(merge(x1e, batch_size=1, padding=2 * padding))
            c = spn.upsample2(x2e)
            low = spn.upsample_lowres(ie_feat)
            low = spn.fuse_lowres(torch.cat((c, low), dim=1))
            return (x_latent0, x_latent1, a, b, low)

    spn.patch_encoder = _FakeViT()
    return Front().eval(), pred


def export_and_inspect(n_patches):
    import onnx

    front, _ = build_front(n_patches)
    path = OUT / f"front_{n_patches}.onnx"
    pe_feat = torch.randn(n_patches, 4, 24, 24)
    pe_i0 = torch.randn(n_patches, 577, 4)
    pe_i1 = torch.randn(n_patches, 577, 4)
    ie_feat = torch.randn(1, 4, 24, 24)
    with torch.no_grad():
        torch.onnx.export(
            front, (pe_feat, pe_i0, pe_i1, ie_feat), str(path),
            opset_version=17,
            input_names=["pe_feat", "pe_int0", "pe_int1", "ie_feat"],
            output_names=["enc0", "enc1", "enc2", "enc3", "enc4"],
            dynamo=False, do_constant_folding=True)

    model = onnx.load(str(path))
    # The legacy tracer emits split sizes as `Constant` nodes (not
    # initializers), so collect from both places.
    const_vals = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                arr = onnx.numpy_helper.to_array(attr.t)
                if arr.ndim == 1 and arr.size <= 4:
                    const_vals[node.output[0]] = arr.tolist()
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        if arr.ndim == 1 and arr.size <= 4:
            const_vals[init.name] = arr.tolist()

    split_sizes = []
    for node in model.graph.node:
        if node.op_type != "Split":
            continue
        for inp in node.input[1:]:
            if inp in const_vals:
                split_sizes.append(const_vals[inp])
    return path, split_sizes


print("=== graph-level check ===")
ok = True
for n_patches, expect in ((35, [25, 9, 1]), (21, [16, 4, 1])):
    path, found = export_and_inspect(n_patches)
    hit = expect in found
    wrong = ([25, 9, 1] if n_patches == 21 else [16, 4, 1]) in found
    print(f"n={n_patches}: Split sizes found={found}")
    print(f"   expect {expect}: {'OK' if hit else 'FAIL (not present)'}"
          f"   stale-other-layout present: {wrong}")
    ok &= hit and not wrong

print("\n=== executability check (21-patch graph on CPU EP) ===")
import onnxruntime as ort

sess = ort.InferenceSession(str(OUT / "front_21.onnx"),
                            providers=["CPUExecutionProvider"])
feeds = {
    "pe_feat": torch.randn(21, 4, 24, 24).numpy(),
    "pe_int0": torch.randn(21, 577, 4).numpy(),
    "pe_int1": torch.randn(21, 577, 4).numpy(),
    "ie_feat": torch.randn(1, 4, 24, 24).numpy(),
}
outs = sess.run(None, feeds)
names = [o.name for o in sess.get_outputs()]
for nm, o in zip(names, outs):
    print(f"   {nm:5s} {tuple(o.shape)}")
# enc2 is the x0 merge output: 4x4 grid of 24x24 => 96x96 spatial.
enc2 = outs[2]
assert enc2.shape[-1] == 96 and enc2.shape[-2] == 96, \
    f"x0 merge should be 96x96 (4x24), got {enc2.shape}"
print("   enc2 96x96 confirms the 4x4 grid merged with padding=0")
print("\nALL GRAPH CHECKS PASSED" if ok else "\nGRAPH CHECKS FAILED")
sys.exit(0 if ok else 1)
