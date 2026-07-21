"""SHARP model loading and compiled inference.

BUG#7 RESOLVED: PyTorch 2.13 fixes the "Python int too large to convert to C long"
    error on Windows. mode="max-autotune" now works correctly.

BUG#8 FIX: CUDA non-default streams are incompatible with torch.compile
    on Windows (Triton limitation → OverflowError). No dual-stream pipeline.

Performance: compile(max-autotune, dynamic=False) + FP16 autocast gives ~6%
    over compile(default), ~1.5x over raw FP32 eager.
"""

import torch
from sharp.models import PredictorParams, create_predictor

MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


class SharpPredictor:
    """Wrapper for SHARP predictor with compile + FP16 optimization."""

    def __init__(self, device: torch.device = torch.device("cuda"),
                 use_compile: bool = True, use_fp16: bool = True):
        self.device = device
        self.use_fp16 = use_fp16
        self._compiled = None

        # Load model
        state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URL, progress=True, map_location="cpu"
        )
        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval().to(device)

        if use_compile:
            # Eliminate graph break from Tensor.item() in GaussianComposer
            torch._dynamo.config.capture_scalar_outputs = True
            self._compiled = torch.compile(
                self.predictor, mode="max-autotune", dynamic=False
            )
        else:
            self._compiled = self.predictor

        self._warmed_up = False

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
