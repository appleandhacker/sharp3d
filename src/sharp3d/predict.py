"""SHARP model loading and compiled inference.

Performance options (in priority order):
    1. ORT TensorRT: patch_encoder via ONNX Runtime TensorRT FP16 (1.74x on ViT)
    2. torch.compile: max-autotune + FP16 autocast (~1.5x over eager)
    3. Eager fallback: no compilation

BUG#7 RESOLVED: PyTorch 2.13 fixes the "Python int too large to convert to C long"
    error on Windows. mode="max-autotune" now works correctly.

BUG#8 FIX: CUDA non-default streams are incompatible with torch.compile
    on Windows (Triton limitation → OverflowError). No dual-stream pipeline.
"""

import logging

import torch
from sharp.models import PredictorParams, create_predictor

logger = logging.getLogger(__name__)

MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


class SharpPredictor:
    """Wrapper for SHARP predictor with ORT TensorRT + compile + FP16 optimization."""

    def __init__(self, device: torch.device = torch.device("cuda"),
                 use_compile: bool = True, use_fp16: bool = True,
                 use_ort: bool = True):
        self.device = device
        self.use_fp16 = use_fp16
        self.use_ort = False
        self._compiled = None

        # Load model
        state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URL, progress=True, map_location="cpu"
        )
        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval().to(device)

        # Try ORT TensorRT acceleration for patch_encoder
        if use_ort:
            self.use_ort = self._setup_ort()

        if use_compile:
            # Eliminate graph break from Tensor.item() in GaussianComposer
            torch._dynamo.config.capture_scalar_outputs = True
            self._compiled = torch.compile(
                self.predictor, mode="max-autotune", dynamic=False
            )
        else:
            self._compiled = self.predictor

        self._warmed_up = False

    def _setup_ort(self) -> bool:
        """Replace patch_encoder with ONNX Runtime TensorRT version.

        Returns True if ORT is active, False if fallback to PyTorch.
        """
        try:
            from .ort_engine import create_ort_patch_encoder

            ort_encoder = create_ort_patch_encoder(self.predictor, self.device)
            if ort_encoder is None:
                return False

            # Replace patch_encoder in the model
            spn = self.predictor.monodepth_model.monodepth_predictor.encoder
            spn.patch_encoder = ort_encoder
            logger.info("patch_encoder replaced with ORT TensorRT (FP16)")
            return True

        except Exception as e:
            logger.warning("ORT setup failed, using PyTorch: %s", e)
            return False

    def warmup(self, img_resized: torch.Tensor, disparity_factor: torch.Tensor):
        """Run one inference to trigger compilation (first call is slow)."""
        if self._warmed_up:
            return
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            _ = self._compiled(img_resized, disparity_factor)
        torch.cuda.synchronize()
        self._warmed_up = True

    @torch.no_grad()
    def predict(self, img_resized: torch.Tensor,
                disparity_factor: torch.Tensor):
        """Run SHARP prediction.

        Args:
            img_resized: (1, 3, 1536, 1536) float tensor.
            disparity_factor: (1,) float32 tensor.

        Returns:
            Gaussians3D in NDC space.
        """
        if self.use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._compiled(img_resized, disparity_factor)
        else:
            return self._compiled(img_resized, disparity_factor)
