"""ONNX Runtime TensorRT engine for SHARP acceleration.

Provides FP16 TensorRT-accelerated inference for the DINOv2 ViT encoders,
replacing PyTorch forward passes with ONNX Runtime + TensorRT EP.

Key optimizations:
    - IO Binding: zero-copy GPU→ORT→GPU (no CPU roundtrip)
    - Pre-allocated output tensors on CUDA
    - Dynamic output shape detection (supports 1 or 5 outputs)
    - Supports both patch_encoder (35/21 patches) and image_encoder (1 patch)

Requirements:
    - onnxruntime-gpu (with TensorrtExecutionProvider)
    - cuDNN 9 DLLs in PATH (found in torch/lib/)
    - ONNX models exported to .cache/onnx/

Performance: patch_encoder 35 patches: 418ms (PyTorch) → 240ms (TRT FP16) = 1.74x
             IO Binding saves ~30ms additional by eliminating CPU roundtrip
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
_TRT_CACHE_DIR = _CACHE_DIR / "trt_v3"  # v3: rebuilt with 2GB workspace


def _ensure_cudnn_path():
    """Add torch/lib to PATH so ONNX Runtime can find cuDNN 9 DLLs."""
    try:
        import torch
        torch_lib = str(Path(torch.__file__).parent / "lib")
        if torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


# ── ONNX export helpers ──────────────────────────────────────────────

def _export_onnx_model(module, onnx_path: Path, device: torch.device,
                      dummy_input: torch.Tensor, input_name: str = "patches",
                      output_names=None, dynamo=True):
    """Export a PyTorch module to ONNX format.

    Args:
        module: PyTorch module to export.
        onnx_path: Output .onnx file path.
        device: CUDA device.
        dummy_input: Example input tensor for tracing.
        input_name: Name for the input.
        output_names: List of output names (auto-detected if None).
        dynamo: Use dynamo-based export (default True). The dynamo exporter
                properly unrolls dict returns into separate ONNX outputs.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()

    if output_names is None:
        with torch.no_grad():
            out = module(dummy_input)
        if isinstance(out, tuple):
            n_out = len(out)
        else:
            n_out = 1
        output_names = [f"output_{i}" for i in range(n_out)]

    logger.info("Exporting to ONNX: %s (outputs: %s, dynamo=%s)",
                onnx_path, output_names, dynamo)
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_input,
            str(onnx_path),
            opset_version=17,
            input_names=[input_name],
            output_names=output_names,
            dynamic_axes={input_name: {0: "batch"}},
            dynamo=dynamo,
        )
    logger.info("ONNX export complete: %.1f MB", onnx_path.stat().st_size / 1e6)


def export_patch_encoder(predictor, onnx_path: Path, device: torch.device):
    """Export patch_encoder to ONNX format."""
    patch_encoder = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder
    dummy = torch.randn(35, 3, 384, 384, device=device, dtype=torch.float32)
    _export_onnx_model(
        patch_encoder, onnx_path, device, dummy,
        output_names=["features", "intermediates", "add_1197", "add_1785", "add_2373"],
    )


def export_image_encoder(predictor, onnx_path: Path, device: torch.device):
    """Export image_encoder to ONNX format.

    Sets intermediate_features_ids before export since the image_encoder
    has it as None by default (the SPN doesn't use image intermediates).
    This ensures the ONNX model has 5 outputs matching the patch_encoder.
    """
    image_encoder = predictor.monodepth_model.monodepth_predictor.encoder.image_encoder
    # Ensure intermediate features are captured for ONNX export
    if getattr(image_encoder, 'intermediate_features_ids', None) is None:
        image_encoder.intermediate_features_ids = [5, 11, 17, 23]
    dummy = torch.randn(1, 3, 384, 384, device=device, dtype=torch.float32)
    _export_onnx_model(
        image_encoder, onnx_path, device, dummy,
        output_names=["features", "intermediates", "add_1197", "add_1785", "add_2373"],
    )


# ── ORT encoder module ───────────────────────────────────────────────

class ORTEncoder(nn.Module):
    """Drop-in replacement for TimmViT encoders using ONNX Runtime TensorRT.

    Uses IO Binding for zero-copy GPU→ORT→GPU inference, eliminating the
    CPU numpy roundtrip that costs ~30ms per batch.

    Dynamically detects output count from the ONNX model, supporting both
    full 5-output encoders (patch_encoder) and single-output encoders.
    """

    # ViT-L/16 at 384 resolution
    _GRID_SIZE = (24, 24)       # 384 / 16 = 24
    _NUM_PREFIX = 1             # CLS token
    _INTERMEDIATE_IDS = [5, 11, 17, 23]

    def __init__(self, onnx_path: Path, trt_cache_dir: Path,
                 device: torch.device, label: str = "encoder",
                 grid_size: tuple = None, intermediate_ids: list = None,
                 int8_enable: bool = False):
        super().__init__()
        self.device = device
        self._onnx_path = onnx_path
        self._trt_cache_dir = trt_cache_dir
        self._label = label
        self._int8_enable = int8_enable

        _ensure_cudnn_path()
        self._session = self._create_session()

        # Cache I/O metadata
        self._input_name = self._session.get_inputs()[0].name
        outputs_meta = self._session.get_outputs()
        self._output_names = [o.name for o in outputs_meta]
        self._n_outputs = len(outputs_meta)

        # Build dynamic shape templates: replace 'batch' string with -1
        self._output_shape_templates = []
        for o in outputs_meta:
            shape = []
            for dim in o.shape:
                if isinstance(dim, str):
                    shape.append(-1)  # placeholder for dynamic dim
                else:
                    shape.append(int(dim))
            self._output_shape_templates.append(tuple(shape))

        # TimmViT-compatible attributes (configurable for speed mode)
        self._grid_size = grid_size or self._GRID_SIZE
        self._num_prefix_tokens = self._NUM_PREFIX
        self.intermediate_features_ids = list(
            intermediate_ids or self._INTERMEDIATE_IDS)

        logger.info("ORT %s ready: %d outputs, grid=%s, int8=%s",
                     label, self._n_outputs, self._grid_size, int8_enable)

    def reshape_feature(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Discard class token and reshape 1D feature map to 2D grid."""
        batch_size, seq_len, channel = embeddings.shape
        height, width = self._grid_size
        if self._num_prefix_tokens:
            embeddings = embeddings[:, self._num_prefix_tokens:, :]
        return embeddings.reshape(batch_size, height, width, channel).permute(0, 3, 1, 2)

    def _create_session(self):
        """Create ONNX Runtime session with TensorRT EP (fallback to CUDA/CPU)."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 2

        self._trt_cache_dir.mkdir(parents=True, exist_ok=True)

        # Check for INT8 calibration table
        calib_table = self._trt_cache_dir / f"{self._label}_calibration_table"
        use_int8 = self._int8_enable and calib_table.exists()
        if self._int8_enable and not calib_table.exists():
            logger.warning("INT8 requested but no calibration table at %s, "
                           "using FP16. Run: python -m sharp3d.calibrate_int8",
                           calib_table)

        trt_opts = {
            "trt_fp16_enable": True,
            # 2GB workspace per engine. 1GB forced TRT into slower algorithms
            # (speed regression); 4GB overflowed 12GB VRAM into shared memory.
            # 2GB keeps total VRAM ~9.5GB (safe margin) with fast algorithms.
            "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(self._trt_cache_dir),
        }
        if use_int8:
            trt_opts["trt_int8_enable"] = True
            trt_opts["trt_int8_calibration_table_name"] = str(calib_table)
            logger.info("INT8 enabled for %s (calibration: %s)",
                         self._label, calib_table)

        providers = [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]

        session = ort.InferenceSession(
            str(self._onnx_path), sess_options=so, providers=providers
        )
        active = session.get_providers()

        if "TensorrtExecutionProvider" in active:
            mode = "INT8" if use_int8 else "FP16"
            logger.info("Using TensorRT %s for %s", mode, self._label)
        elif "CUDAExecutionProvider" in active:
            logger.info("TensorRT unavailable, using CUDA EP for %s", self._label)
        else:
            logger.warning("GPU EPs unavailable for %s, using CPU (slow!)", self._label)

        return session

    def _resolve_shape(self, template, batch):
        """Replace dynamic dimension placeholder with actual batch size."""
        return tuple(batch if d == -1 else d for d in template)

    def _forward_iobinding(self, x: torch.Tensor):
        """Zero-copy IO Binding path: GPU tensor → ORT → GPU tensors."""
        batch = x.shape[0]
        # Ensure FP32: after predictor.half(), patches may arrive as FP16,
        # but the ONNX model and IO Binding expect FP32 input.
        x_contig = x.detach().float().contiguous()

        # Pre-allocate output tensors on CUDA based on session metadata
        out_tensors = []
        for template in self._output_shape_templates:
            shape = self._resolve_shape(template, batch)
            t = torch.empty(shape, dtype=torch.float32, device=self.device)
            out_tensors.append(t)

        # Create IO binding
        binding = self._session.io_binding()

        # Bind input (GPU tensor directly, no CPU transfer)
        binding.bind_input(
            name=self._input_name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(x_contig.shape),
            buffer_ptr=x_contig.data_ptr(),
        )

        # Bind outputs to pre-allocated CUDA tensors
        for name, t in zip(self._output_names, out_tensors):
            binding.bind_output(
                name=name,
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(t.shape),
                buffer_ptr=t.data_ptr(),
            )

        # Run inference — data stays on GPU the entire time
        self._session.run_with_iobinding(binding)

        # Construct return values matching TimmViT interface
        features = out_tensors[0]
        intermediates = {}
        n_intermediates = self._n_outputs - 1
        for i in range(min(n_intermediates, len(self.intermediate_features_ids))):
            block_id = self.intermediate_features_ids[i]
            intermediates[block_id] = out_tensors[i + 1]

        return features, intermediates

    def _forward_numpy(self, x: torch.Tensor):
        """Fallback path: GPU → CPU numpy → ORT → CPU numpy → GPU."""
        x_np = x.detach().cpu().numpy()
        outputs = self._session.run(None, {self._input_name: x_np})

        features = torch.from_numpy(outputs[0]).to(self.device, non_blocking=True)
        intermediates = {}
        n_intermediates = len(outputs) - 1
        for i in range(min(n_intermediates, len(self.intermediate_features_ids))):
            block_id = self.intermediate_features_ids[i]
            intermediates[block_id] = torch.from_numpy(
                outputs[i + 1]).to(self.device, non_blocking=True)

        return features, intermediates

    @torch.compiler.disable(recursive=False)
    def forward(self, x: torch.Tensor):
        """Run encoder via ONNX Runtime.

        Uses IO Binding (zero-copy GPU path) when available, with fallback
        to numpy transfer for older ORT versions.

        Args:
            x: [batch, 3, 384, 384] float tensor on CUDA.

        Returns:
            (features, intermediates) matching TimmViT output format:
                features: [batch, 1024, 24, 24]
                intermediates: dict {block_id: [batch, 577, 1024]}
                               (empty if ONNX model has 1 output)
        """
        # torch.compile launches the patch-producing kernels asynchronously on
        # PyTorch's stream, but ORT runs inference on its own separate stream.
        # Without this full sync, ORT can read patches that are not yet fully
        # produced → corrupted ViT features → intermittent blurry frames.
        torch.cuda.synchronize()
        try:
            return self._forward_iobinding(x)
        except Exception as e:
            if not getattr(self, '_iobinding_failed', False):
                logger.warning("IO Binding failed, falling back to numpy: %s", e)
                self._iobinding_failed = True
            return self._forward_numpy(x)


# ── Factory functions ────────────────────────────────────────────────

def create_ort_patch_encoder(
    predictor,
    device: torch.device,
    onnx_dir: Path = _ONNX_DIR,
    trt_cache_dir: Path = _TRT_CACHE_DIR,
    int8_enable: bool = False,
) -> ORTEncoder | None:
    """Create an ORT-accelerated patch_encoder, exporting ONNX if needed."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.warning("onnxruntime not installed, skipping ORT acceleration")
        return None

    onnx_path = onnx_dir / "patch_encoder.onnx"
    if not onnx_path.exists():
        try:
            export_patch_encoder(predictor, onnx_path, device)
        except Exception as e:
            logger.warning("ONNX export failed: %s", e)
            return None

    try:
        return ORTEncoder(onnx_path, trt_cache_dir, device,
                          label="patch_encoder", int8_enable=int8_enable)
    except Exception as e:
        logger.warning("ORT session creation failed: %s", e)
        return None


def create_ort_image_encoder(
    predictor,
    device: torch.device,
    onnx_dir: Path = _ONNX_DIR,
    trt_cache_dir: Path = _TRT_CACHE_DIR,
    int8_enable: bool = False,
) -> ORTEncoder | None:
    """Create an ORT-accelerated image_encoder, exporting ONNX if needed."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None

    onnx_path = onnx_dir / "image_encoder.onnx"
    # Re-export if the model doesn't have all 5 outputs
    need_export = not onnx_path.exists()
    if not need_export:
        try:
            import onnxruntime as ort
            _sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])
            need_export = len(_sess.get_outputs()) < 5
        except Exception:
            need_export = True

    if need_export:
        try:
            export_image_encoder(predictor, onnx_path, device)
        except Exception as e:
            logger.warning("Image encoder ONNX export failed: %s", e)
            return None

    try:
        return ORTEncoder(onnx_path, trt_cache_dir, device,
                          label="image_encoder", int8_enable=int8_enable)
    except Exception as e:
        logger.warning("Image encoder ORT session creation failed: %s", e)
        return None
