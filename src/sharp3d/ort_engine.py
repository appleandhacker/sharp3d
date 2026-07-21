"""ONNX Runtime TensorRT engine for SHARP patch_encoder acceleration.

Provides FP16 TensorRT-accelerated inference for the DINOv2 ViT patch encoder,
replacing the PyTorch forward pass with ONNX Runtime + TensorRT EP.

Requirements:
    - onnxruntime-gpu (with TensorrtExecutionProvider)
    - cuDNN 9 DLLs in PATH (found in torch/lib/)
    - ONNX model exported to .cache/onnx/patch_encoder.onnx

Performance: patch_encoder 35 patches: 418ms (PyTorch) → 240ms (TensorRT FP16) = 1.74x
"""

import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default cache directories (relative to project root)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
_ONNX_DIR = _CACHE_DIR / "onnx"
_TRT_CACHE_DIR = _CACHE_DIR / "trt"


def _ensure_cudnn_path():
    """Add torch/lib to PATH so ONNX Runtime can find cuDNN 9 DLLs."""
    try:
        import torch
        torch_lib = str(Path(torch.__file__).parent / "lib")
        if torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def export_onnx_model(predictor, onnx_path: Path, device: torch.device):
    """Export patch_encoder to ONNX format.

    Args:
        predictor: Loaded SHARP predictor (create_predictor output).
        onnx_path: Output .onnx file path.
        device: CUDA device.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    patch_encoder = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder
    patch_encoder.eval()

    dummy = torch.randn(35, 3, 384, 384, device=device, dtype=torch.float32)

    logger.info("Exporting patch_encoder to ONNX: %s", onnx_path)
    with torch.no_grad():
        torch.onnx.export(
            patch_encoder,
            dummy,
            str(onnx_path),
            opset_version=17,
            input_names=["patches"],
            output_names=["features", "intermediates", "add_1197", "add_1785", "add_2373"],
            dynamic_axes={"patches": {0: "batch"}, "features": {0: "batch"}},
        )
    logger.info("ONNX export complete: %.1f MB", onnx_path.stat().st_size / 1e6)


class ORTPatchEncoder(nn.Module):
    """Drop-in replacement for TimmViT patch_encoder using ONNX Runtime TensorRT.

    Runs the DINOv2 ViT forward pass via ONNX Runtime with TensorRT FP16,
    returning the same (features, intermediates) tuple as the original module.
    """

    def __init__(self, onnx_path: Path, trt_cache_dir: Path, device: torch.device):
        super().__init__()
        self.device = device
        self._session = None
        self._onnx_path = onnx_path
        self._trt_cache_dir = trt_cache_dir

        # Ensure cuDNN is findable
        _ensure_cudnn_path()

        # Create session
        self._session = self._create_session()

        # TimmViT-compatible attributes for reshape_feature
        self._grid_size = (24, 24)  # 384 / patch_size(16) = 24
        self._num_prefix_tokens = 1  # CLS token
        self.intermediate_features_ids = [5, 11, 17, 23]

    def reshape_feature(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Discard class token and reshape 1D feature map to 2D grid.

        Matches TimmViT.reshape_feature interface.
        """
        batch_size, seq_len, channel = embeddings.shape
        height, width = self._grid_size
        # Remove CLS token
        if self._num_prefix_tokens:
            embeddings = embeddings[:, self._num_prefix_tokens:, :]
        # [batch, h*w, dim] -> [batch, dim, h, w]
        return embeddings.reshape(batch_size, height, width, channel).permute(0, 3, 1, 2)

    def _create_session(self):
        """Create ONNX Runtime session with TensorRT EP (fallback to CUDA/CPU)."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 2  # warnings only

        # Ensure TRT cache dir exists
        self._trt_cache_dir.mkdir(parents=True, exist_ok=True)

        # Provider priority: TensorRT > CUDA > CPU
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(self._trt_cache_dir),
            }),
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]

        session = ort.InferenceSession(
            str(self._onnx_path), sess_options=so, providers=providers
        )
        active = session.get_providers()
        logger.info("ORT providers: %s", active)

        if "TensorrtExecutionProvider" in active:
            logger.info("Using TensorRT FP16 for patch_encoder")
        elif "CUDAExecutionProvider" in active:
            logger.info("TensorRT unavailable, using CUDA EP")
        else:
            logger.warning("GPU EPs unavailable, using CPU (slow!)")

        return session

    @torch.compiler.disable(recursive=False)
    def forward(self, x: torch.Tensor):
        """Run patch_encoder via ONNX Runtime.

        Args:
            x: [batch, 3, 384, 384] float tensor on CUDA.

        Returns:
            (features, intermediates) matching TimmViT output format:
                features: [batch, 1024, 24, 24]
                intermediates: list of 4 × [batch, 577, 1024]
        """
        # GPU → CPU numpy (fast: ~2ms for 35×3×384×384)
        x_np = x.detach().cpu().numpy()

        # Run ORT inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: x_np})

        # CPU numpy → GPU torch
        features = torch.from_numpy(outputs[0]).to(self.device, non_blocking=True)
        intermediates = {
            block_id: torch.from_numpy(outputs[i + 1]).to(self.device, non_blocking=True)
            for i, block_id in enumerate(self.intermediate_features_ids)
        }

        return features, intermediates


def create_ort_patch_encoder(
    predictor,
    device: torch.device,
    onnx_dir: Path = _ONNX_DIR,
    trt_cache_dir: Path = _TRT_CACHE_DIR,
) -> ORTPatchEncoder | None:
    """Create an ORT-accelerated patch_encoder, exporting ONNX if needed.

    Args:
        predictor: Loaded SHARP predictor.
        device: CUDA device.
        onnx_dir: Directory for ONNX model files.
        trt_cache_dir: Directory for TensorRT engine cache.

    Returns:
        ORTPatchEncoder instance, or None if ORT is unavailable.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.warning("onnxruntime not installed, skipping ORT acceleration")
        return None

    onnx_path = onnx_dir / "patch_encoder.onnx"

    # Export ONNX model if not cached
    if not onnx_path.exists():
        try:
            export_onnx_model(predictor, onnx_path, device)
        except Exception as e:
            logger.warning("ONNX export failed: %s", e)
            return None

    # Create ORT session
    try:
        return ORTPatchEncoder(onnx_path, trt_cache_dir, device)
    except Exception as e:
        logger.warning("ORT session creation failed: %s", e)
        return None
