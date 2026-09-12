"""Full-model ONNX export (ViTs + SPN + decoder + composer in one graph).

Rationale (measured 2026-09-09): predict() spends ~410ms/frame, split evenly
between the two DINOv2 ViTs (already TRT) and the SPN merge/upsample chain +
multires decoder + gaussian head (torch.compile, ~190ms). Exporting the WHOLE
predictor as a single static graph lets TensorRT fuse across the SPN boundary
and apply INT8 QDQ to the ViT MatMuls. sharp's split/merge are fx.wrapped
atomic ops and the ViT captures intermediates without hooks, so the graph
traces cleanly (verified: 5313 nodes, weights via external data files).

The export runs in FP32 (quantize_static requires an fp32 base graph); TRT
applies FP16/INT8 precision afterwards. Weights >2GB serialize as external
data files next to the .onnx — keep them together.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from sharp.models.encoders.spn_encoder import merge

logger = logging.getLogger(__name__)


# SlidingPyramidNetwork patch layouts, keyed by total patch count.
#   quality (use_patch_overlap=True):  overlap .25/.5 -> x0 5x5=25, x1 3x3=9,
#                                      x2 1x1 -> 35 patches, merge padding 3
#   speed   (use_patch_overlap=False): overlap 0/0    -> x0 4x4=16, x1 2x2=4,
#                                      x2 1x1 -> 21 patches, merge padding 0
# Source of truth: ml-sharp/src/sharp/models/encoders/spn_encoder.py:forward
# (steps 1/3), which selects the overlap ratios and padding from
# use_patch_overlap. Keep this in sync if that ever gains a third mode.
_SPN_LAYOUTS: dict[int, tuple[int, int]] = {35: (5, 3), 21: (4, 2)}


def _spn_geometry(n_patches: int) -> tuple[int, list[int], int]:
    """Derive the SPN patch geometry from the total patch count.

    Returns ``(x0_tile_size, split_sizes, padding)``. The split sizes drive
    ``torch.split`` on the patch-ViT output and the ``[:x0_tile_size]`` slice
    that feeds the two latent branches; ``padding`` must match, because
    ``merge()`` infers its own grid from ``sqrt(len)`` and then trims each
    interior seam by exactly this many pixels.

    Raises ValueError for an unsupported count rather than exporting a graph
    with the wrong grouping — a mis-grouped graph loads fine and only fails
    (or silently produces wrong geometry) at inference time.
    """
    try:
        g0, g1 = _SPN_LAYOUTS[n_patches]
    except KeyError:
        raise ValueError(
            f"n_patches={n_patches} 不受支持（仅 35=quality / 21=speed）"
        ) from None
    split_sizes = [g0 * g0, g1 * g1, 1]
    assert sum(split_sizes) == n_patches, (split_sizes, n_patches)
    padding = 3 if g0 == 5 else 0
    return split_sizes[0], split_sizes, padding


class _ExportWrapper(nn.Module):
    """Flatten the Gaussians3D NamedTuple into 5 ordered tensor outputs."""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image, disparity_factor):
        g = self.predictor(image, disparity_factor)
        return (g.mean_vectors, g.singular_values, g.quaternions,
                g.colors, g.opacities)


def export_full_model(predictor, onnx_path: Path, device: torch.device,
                      n_patches: int) -> Path:
    """Export the full RGBGaussianPredictor to a static-shape ONNX graph.

    Args:
        predictor: RGBGaussianPredictor (weights may be FP16 — the export
                   graph itself is precision-agnostic; FP32 checkpoint values
                   produce an FP32 base graph as quantize_static expects).
        onnx_path: Output path (.onnx). External weight data lands beside it.
        device: CUDA device for the tracing pass.
        n_patches: 35 (quality) or 21 (speed) — fixes the internal batch dim.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _ExportWrapper(predictor).eval().float()

    image = torch.randn(1, 3, 1536, 1536, device=device)
    df = torch.tensor([1.0], device=device, dtype=torch.float32)

    logger.info("整模型 ONNX 导出开始: %s", onnx_path)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (image, df),
            str(onnx_path),
            opset_version=17,
            input_names=["image", "disparity_factor"],
            output_names=["mean_vectors", "singular_values", "quaternions",
                          "colors", "opacities"],
            dynamo=False,       # legacy tracer: split/merge loops need it
            do_constant_folding=True,
        )
    logger.info("整模型 ONNX 导出完成: %.0fMB", onnx_path.stat().st_size / 2**20)
    return onnx_path


def export_spn_tail(predictor, onnx_path: Path, device: torch.device,
                    n_patches: int) -> Path:
    """Export the post-ViT graph (SPN merge/upsample + decoder + composer).

    Inputs: image (1,3,1536,1536), disparity_factor (1,), patch-ViT features
    (n_patches,1024,24,24), two patch-ViT intermediates (n_patches,577,1024
    each), image-ViT features (1,1024,24,24). Weights ≈0.3GB — single-file ONNX.

    ``n_patches`` (35 quality / 21 speed) fixes both the dummy input shapes and
    the merge grouping — the two are the same table (see ``_spn_geometry``).
    """
    from .spn_tail import SpnTail

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    tail = SpnTail(predictor, n_patches).eval().float()

    image = torch.randn(1, 3, 1536, 1536, device=device)
    df = torch.tensor([1.0], device=device, dtype=torch.float32)
    pe_feat = torch.randn(n_patches, 1024, 24, 24, device=device)
    pe_i0 = torch.randn(n_patches, 577, 1024, device=device)
    pe_i1 = torch.randn(n_patches, 577, 1024, device=device)
    ie_feat = torch.randn(1, 1024, 24, 24, device=device)
    args = (image, df, pe_feat, pe_i0, pe_i1, ie_feat)

    logger.info("SPN 尾部 ONNX 导出开始: %s", onnx_path)
    with torch.no_grad():
        torch.onnx.export(
            tail,
            args,
            str(onnx_path),
            opset_version=17,
            input_names=["image", "disparity_factor", "pe_feat",
                         "pe_int0", "pe_int1", "ie_feat"],
            output_names=["mean_vectors", "singular_values", "quaternions",
                          "colors", "opacities"],
            dynamo=False,
            do_constant_folding=True,
        )
    logger.info("SPN 尾部 ONNX 导出完成: %.0fMB", onnx_path.stat().st_size / 2**20)
    return onnx_path


class _SpnFront(torch.nn.Module):
    """SPN merge/upsample/fuse only — 5 encoding maps from ViT outputs.

    Pure convs + ConvTranspose + slice/cat (no norm layers): measured fp16
    safe, unlike the decoder half whose GroupNorm-family ops degrade under
    TRT fp16 (autocast keeps those in fp32; TRT has no such policy).

    ``n_patches`` selects the patch grouping (35 quality / 21 speed). The
    derived values are plain Python ints/lists stored on the module: the
    legacy (``dynamo=False``) tracer evaluates attribute reads eagerly and
    bakes them in as constants, exactly like ``_TailFromEncodings`` already
    does with ``_n_enc``/``_sort2``. Do NOT turn them into buffers or tensors —
    that would promote them to graph tensors.
    """

    _OUT_NAMES = ["enc0", "enc1", "enc2", "enc3", "enc4"]

    def __init__(self, predictor, n_patches: int):
        super().__init__()
        self.predictor = predictor
        # Validate here, not in forward(): torch.onnx.export wraps exceptions
        # raised during tracing, which would bury this message.
        x0, split_sizes, padding = _spn_geometry(n_patches)
        self._n_patches = n_patches
        self._x0_tile_size = x0        # int → traced constant
        self._split_sizes = split_sizes  # list[int] → onnx::Constant for Split
        self._padding = padding        # int → traced constant

    def forward(self, pe_feat, pe_int0, pe_int1, ie_feat):
        spn = self.predictor.monodepth_model.monodepth_predictor.encoder
        # Latent branches consume only the x0-resolution patches, i.e. the
        # first `batch_size * x0_tile_size` rows (batch_size == 1 here).
        x_latent0 = spn.upsample_latent0(
            merge(spn.patch_encoder.reshape_feature(pe_int0)[:self._x0_tile_size],
                  batch_size=1, padding=self._padding))
        x_latent1 = spn.upsample_latent1(
            merge(spn.patch_encoder.reshape_feature(pe_int1)[:self._x0_tile_size],
                  batch_size=1, padding=self._padding))
        x0e, x1e, x2e = torch.split(pe_feat, self._split_sizes, dim=0)
        x0 = spn.upsample0(merge(x0e, batch_size=1, padding=self._padding))
        x1 = spn.upsample1(merge(x1e, batch_size=1, padding=2 * self._padding))
        x2 = spn.upsample2(x2e)
        lowres = spn.upsample_lowres(ie_feat)
        lowres = spn.fuse_lowres(torch.cat((x2, lowres), dim=1))
        return (x_latent0, x_latent1, x0, x1, lowres)


class _TailFromEncodings(torch.nn.Module):
    """decoder + head + init + feature_model + composer from 5 encodings."""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        adaptor = predictor.monodepth_model
        self._n_enc = len(adaptor.monodepth_predictor.encoder.dims_encoder)
        self._ret_enc = bool(adaptor.return_encoder_features)
        self._ret_dec = bool(adaptor.return_decoder_features)
        self._sort2 = bool(adaptor.num_monodepth_layers == 2
                           and adaptor.sorting_monodepth)

    def forward(self, image, disparity_factor, enc0, enc1, enc2, enc3, enc4):
        pred = self.predictor
        md = pred.monodepth_model.monodepth_predictor
        encoder_features = [enc0, enc1, enc2, enc3, enc4]
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


def export_spn_split(predictor, device: torch.device, n_patches: int,
                     onnx_dir: Path) -> tuple[Path, Path]:
    """Export the two-piece tail: fp16-safe SPN front + fp32 rest.

    Runs the SPN merge/upsample chain (fp16 TRT) separately from the
    decoder + gaussian head + composer (fp32 CUDA EP) — this reproduces
    torch autocast's selective precision, which a single fp16 TRT engine
    cannot express.
    """
    from sharp.utils.gaussians import Gaussians3D  # noqa: F401
    from sharp3d.spn_tail import ViTCapture

    onnx_dir.mkdir(parents=True, exist_ok=True)
    front_path = onnx_dir / f"spn_front_{n_patches}.onnx"
    rest_path = onnx_dir / f"tail_rest_{n_patches}.onnx"

    front = _SpnFront(predictor, n_patches).eval().float()
    rest = _TailFromEncodings(predictor).eval().float()

    pe_feat = torch.randn(n_patches, 1024, 24, 24, device=device)
    pe_i0 = torch.randn(n_patches, 577, 1024, device=device)
    pe_i1 = torch.randn(n_patches, 577, 1024, device=device)
    ie_feat = torch.randn(1, 1024, 24, 24, device=device)

    with torch.no_grad():
        torch.onnx.export(
            front, (pe_feat, pe_i0, pe_i1, ie_feat), str(front_path),
            opset_version=17,
            input_names=["pe_feat", "pe_int0", "pe_int1", "ie_feat"],
            output_names=_SpnFront._OUT_NAMES,
            dynamo=False, do_constant_folding=True)
        encs = front(pe_feat, pe_i0, pe_i1, ie_feat)
        image = torch.randn(1, 3, 1536, 1536, device=device)
        df = torch.tensor([1.0], device=device, dtype=torch.float32)
        torch.onnx.export(
            rest, (image, df, *encs), str(rest_path),
            opset_version=17,
            input_names=["image", "disparity_factor", "enc0", "enc1",
                         "enc2", "enc3", "enc4"],
            output_names=["mean_vectors", "singular_values", "quaternions",
                          "colors", "opacities"],
            dynamo=False, do_constant_folding=True)
    logger.info("SPN 拆分导出完成: front %.0fMB, rest %.0fMB",
                front_path.stat().st_size / 2**20,
                rest_path.stat().st_size / 2**20)
    return front_path, rest_path
