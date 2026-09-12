"""SPN tail: everything AFTER the two ViT encoders as one exportable module.

The full-graph export (single ONNX for the whole predictor) failed at INT8
calibration — the QDQ-augmented patch encoder's transient tensors OOM a 12GB
GPU (24 blocks × ~1.6GB of quantize/dequantize copies of 82-330MB activations,
co-resident with the rest of the graph). The 3-session split avoids it:

    [pyramid+split (torch glue)] → patch ViT (INT8 QDQ TRT)
                                 → image ViT (INT8 QDQ TRT)
                                 → SpnTail (FP16 TRT): merge/upsample +
                                   decoder + head + init + feature_model +
                                   composer

SpnTail replicates SlidingPyramidNetwork.forward steps 3+ and
RGBGaussianPredictor.forward verbatim, but takes the ViT outputs as graph
inputs instead of calling the encoders. Unused submodules (the ViTs) are not
traced, so they don't land in the ONNX graph.

Numerical equivalence with the torch path is verified in
tests/verify_spn_tail.py (fp32, expect max diff ≤1e-4).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sharp.models.encoders.spn_encoder import merge


class SpnTail(nn.Module):
    """ViT outputs in → Gaussians3D fields out (as 5 tensors)."""

    _OUT_FIELDS = ("mean_vectors", "singular_values", "quaternions",
                   "colors", "opacities")

    def __init__(self, predictor, n_patches: int):
        super().__init__()
        self.predictor = predictor
        adaptor = predictor.monodepth_model
        self._n_enc = len(adaptor.monodepth_predictor.encoder.dims_encoder)
        # Patch grouping follows the perf mode (35 quality / 21 speed). Derived
        # in __init__ from the shared table so the exported graph can never
        # disagree with the tensor shapes the caller feeds (see
        # full_export._spn_geometry).
        from .full_export import _spn_geometry
        x0, split_sizes, padding = _spn_geometry(n_patches)
        self._x0_tile_size = x0
        self._split_sizes = split_sizes
        self._padding = padding
        # Traced as constants (same values the adaptor bakes in).
        self._ret_enc = bool(adaptor.return_encoder_features)
        self._ret_dec = bool(adaptor.return_decoder_features)
        self._sort2 = bool(adaptor.num_monodepth_layers == 2
                           and adaptor.sorting_monodepth)

    def forward(self, image, disparity_factor,
                pe_feat, pe_int0, pe_int1, ie_feat):
        pred = self.predictor
        md = pred.monodepth_model.monodepth_predictor
        spn = md.encoder

        # ── SlidingPyramidNetwork.forward steps 3+ (merging) ────────────
        # Latent features come from patch-encoder intermediates
        # (n_patches,577,1024) — strip CLS, reshape to 24×24, merge the
        # x0-resolution grid (5×5 with overlap, 4×4 without).
        # reshape_feature lives on TimmViT; calling the method does not trace
        # the ViT weights into the graph (no forward call).
        x_latent0 = spn.upsample_latent0(
            merge(spn.patch_encoder.reshape_feature(pe_int0)[:self._x0_tile_size],
                  batch_size=1, padding=self._padding))
        x_latent1 = spn.upsample_latent1(
            merge(spn.patch_encoder.reshape_feature(pe_int1)[:self._x0_tile_size],
                  batch_size=1, padding=self._padding))

        # Batch split back into the pyramid's three grids (25+9+1 or 16+4+1).
        x0e, x1e, x2e = torch.split(pe_feat, self._split_sizes, dim=0)
        x0 = spn.upsample0(merge(x0e, batch_size=1, padding=self._padding))
        x1 = spn.upsample1(merge(x1e, batch_size=1, padding=2 * self._padding))
        x2 = spn.upsample2(x2e)

        lowres = spn.upsample_lowres(ie_feat)
        lowres = spn.fuse_lowres(torch.cat((x2, lowres), dim=1))

        encoder_features = [x_latent0, x_latent1, x0, x1, lowres]

        # ── MonodepthWithEncodingAdaptor.forward ────────────────────────
        decoder_features = md.decoder(encoder_features[:self._n_enc])
        disparity = md.head(decoder_features)
        if self._sort2:
            first = disparity.max(dim=1, keepdims=True).values
            second = disparity.min(dim=1, keepdims=True).values
            disparity = torch.cat([first, second], dim=1)

        output_features = []
        if self._ret_enc:
            output_features.extend(encoder_features)
        if self._ret_dec:
            output_features.append(decoder_features)

        # ── RGBGaussianPredictor.forward ────────────────────────────────
        df = disparity_factor[:, None, None, None]
        monodepth = df / disparity.clamp(min=1e-4, max=1e4)
        monodepth, _ = pred.depth_alignment(monodepth, None, decoder_features)

        init_output = pred.init_model(image, monodepth)
        image_features = pred.feature_model(
            init_output.feature_input, encodings=output_features)
        delta = pred.prediction_head(image_features)
        g = pred.gaussian_composer(
            delta=delta, base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale)
        return (g.mean_vectors, g.singular_values, g.quaternions,
                g.colors, g.opacities)


class ViTCapture(nn.Module):
    """Transparent wrapper that records a ViT's outputs for verification."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.last = None

    def __getattr__(self, name):
        # Delegate non-module attributes (reshape_feature, ids, ...) to the
        # wrapped ViT so spn.forward sees an exact TimmViT interface.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is None:
                # _modules["inner"] missing raised a bare KeyError before
                # (masking the real AttributeError at e.g. copy.deepcopy or
                # pickling time, when __getattr__ runs before __init__).
                raise AttributeError(
                    f"{type(self).__name__!r} object has no attribute {name!r}"
                    f" (no inner module is set)") from None
            return getattr(inner, name)

    def forward(self, x):
        out = self.inner(x)
        self.last = out
        return out


def capture_vit_outputs(predictor, image: torch.Tensor,
                        disparity_factor: torch.Tensor):
    """Run the full torch predictor once, capturing ViT in/out tensors.

    Returns (gaussians, dict with patch/image ViT tensors) for tail
    verification and for building rest-graph test vectors.
    """
    spn = predictor.monodepth_model.monodepth_predictor.encoder
    pe_wrap = ViTCapture(spn.patch_encoder).to(next(spn.parameters()).device)
    ie_wrap = ViTCapture(spn.image_encoder).to(next(spn.parameters()).device)
    spn.patch_encoder, spn.image_encoder = pe_wrap, ie_wrap
    try:
        with torch.no_grad():
            g = predictor(image, disparity_factor)
    finally:
        spn.patch_encoder = pe_wrap.inner
        spn.image_encoder = ie_wrap.inner
    pe_feat, pe_ints = pe_wrap.last
    ie_feat, _ = ie_wrap.last
    ids = spn.patch_intermediate_features_ids
    return g, {
        "pe_feat": pe_feat,
        "pe_int0": pe_ints[ids[0]],
        "pe_int1": pe_ints[ids[1]],
        "ie_feat": ie_feat,
    }
