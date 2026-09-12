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
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import profiling

logger = logging.getLogger(__name__)


def _log_ort_error(msg: str) -> None:
    """Write ORT error to log file (visible even without logging handler)."""
    logger.warning(msg)
    try:
        log = Path(os.environ.get("LOCALAPPDATA", ".")) / "sharp3d" / "ort_error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

# Default cache directories. Must resolve to the same place as predict.py and
# gui/worker.py — in a frozen build __file__ points inside the (possibly
# read-only) install dir, which silently defeats the TRT engine cache.
from . import resolve_cache_dir as _resolve_cache_dir

_CACHE_DIR = _resolve_cache_dir()
_ONNX_DIR = _CACHE_DIR / "onnx"
_TRT_CACHE_DIR = _CACHE_DIR / "trt_v3"  # v3: rebuilt with 2GB workspace


def _ascii_safe_trt_dir(base: Path) -> Path:
    """TRT EP creates/reads its engine cache via narrow-char CRT APIs.

    A non-ASCII install path (e.g. ``D:\\桌面\\sharp3d``) becomes mojibake
    inside ORT and directory creation fails with 'The system cannot find the
    path specified'. Redirect engine caching to ProgramData (guaranteed
    ASCII, user-writable) when the install path is not ASCII-safe. Engine
    filenames carry the SM tag, so 40- and 50-series machines can share this
    location.
    """
    if str(base).isascii():
        return base
    fb = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "sharp3d" / "trt_cache"
    fb.mkdir(parents=True, exist_ok=True)
    logger.info("TRT 缓存路径含非 ASCII 字符，引擎缓存改用 %s", fb)
    return fb


def _ensure_cudnn_path():
    """Make ONNX Runtime's TRT provider DLLs resolvable on clean machines.

    Two directories matter and neither is on a stock machine's search path:
    - torch/lib for cuDNN/cuBLAS/cudart (nvinfer's runtime dependencies);
    - site-packages/tensorrt_libs for nvinfer_10.dll itself. On a dev box
      that also has the CUDA Toolkit installed these come from CUDA's lib
      dir on PATH, which masks the problem until the app is deployed to a
      machine without the toolkit (Error 126: nvinfer_10.dll is missing).
    """
    try:
        import torch
        torch_lib = str(Path(torch.__file__).parent / "lib")
        if torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(torch_lib)
        except (AttributeError, OSError):
            pass
        # TensorRT 10 pip wheel (tensorrt_libs) ships nvinfer_10.dll etc.
        try:
            import importlib.util
            spec = importlib.util.find_spec("tensorrt_libs")
            if spec and spec.submodule_search_locations:
                trt_dir = Path(list(spec.submodule_search_locations)[0])
                if trt_dir.is_dir():
                    trt_s = str(trt_dir)
                    if trt_s not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = trt_s + os.pathsep + os.environ.get("PATH", "")
                    try:
                        os.add_dll_directory(trt_dir)
                    except (AttributeError, OSError):
                        pass
        except (ImportError, ValueError):
            pass
    except Exception:
        pass


# ── ONNX export helpers ──────────────────────────────────────────────

def onnx_models_cached() -> bool:
    """True when both encoder ONNX files are already on disk.

    Mirrors the resolution order used by create_ort_*_encoder (bundled
    _MEIPASS/models first in frozen builds, .cache/onnx otherwise). When this
    is True the predictor never needs FP32 weights, so predict.py can build
    the model directly in FP16 and skip materializing a transient 2.8GB FP32
    copy (which also halves the weights' H2D transfer).
    """
    import sys as _sys
    for name in ("patch_encoder.onnx", "image_encoder.onnx"):
        if getattr(_sys, "frozen", False):
            p = Path(_sys._MEIPASS) / "models" / name
        else:
            p = _ONNX_DIR / name
        if not p.exists():
            return False
    return True


def _export_onnx_model(module, onnx_path: Path, device: torch.device,
                      dummy_input: torch.Tensor, input_name: str = "patches",
                      output_names=None, dynamo=None):
    """Export a PyTorch module to ONNX format.

    Args:
        module: PyTorch module to export.
        onnx_path: Output .onnx file path.
        device: CUDA device.
        dummy_input: Example input tensor for tracing.
        input_name: Name for the input.
        output_names: List of output names (auto-detected if None).
        dynamo: Use dynamo-based export. None = auto (False in frozen builds).
    """
    import sys as _sys
    if dynamo is None:
        dynamo = not getattr(_sys, "frozen", False)
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
    """Export patch_encoder to ONNX format.

    The batch dim is fixed at 35 here deliberately. Unlike spn_front/tail_rest
    (which are per-mode files), ``patch_encoder.onnx`` is a single shared
    artifact whose cached TRT engine is reused across both perf modes, so it
    must carry the quality-mode shape; the speed-mode runtime feeds 21 patches
    and TRT rebuilds/re-optimizes as needed. Exporting this at 21 would break
    the quality path.
    """
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
                 qdq: bool = False,
                 workspace_gb: int = 2):
        super().__init__()
        self.device = device
        self._onnx_path = onnx_path
        self._trt_cache_dir = trt_cache_dir
        self._label = label
        self._qdq = qdq   # explicit QDQ graph (FP8 experiment)
        # Honored in _create_session below (it used to be accepted and then
        # silently ignored — the workspace size stayed hardcoded at 2GB while
        # the parameter implied otherwise).
        self._workspace_gb = int(workspace_gb)

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

        # IO Binding failure bookkeeping. A *transient* failure (most commonly
        # a CUDA OOM while the video pipeline holds its double buffers) must not
        # disable the zero-copy path for the rest of the session: that turned a
        # one-frame hiccup into a permanent ~30ms/frame numpy penalty with no
        # visible indication. Only a run of consecutive failures is treated as
        # structural (unsupported by this ORT build) and latches the fallback.
        self._iobinding_failed = False
        self._iobinding_fail_streak = 0
        self._IOBINDING_MAX_FAILS = 5

        self.using_trt = "TensorrtExecutionProvider" in             self._session.get_providers()
        logger.info("ORT %s ready: %d outputs, grid=%s, trt=%s",
                     label, self._n_outputs, self._grid_size, self.using_trt)

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

        cache_base = _ascii_safe_trt_dir(self._trt_cache_dir)
        if self._workspace_gb != 2:
            # Engines built with a different trt_max_workspace_size can pick
            # different algorithms, so they must not share the default cache
            # dir (a stale engine for another workspace would be reused
            # silently). The default (2GB) keeps the historical path so all
            # existing caches stay valid — no forced rebuild for anyone.
            cache_base = cache_base / f"ws{self._workspace_gb}"
        cache_base.mkdir(parents=True, exist_ok=True)

        trt_opts = {
            "trt_fp16_enable": True,
            # 2GB workspace per engine. 1GB forced TRT into slower algorithms
            # (speed regression); 4GB overflowed 12GB VRAM into shared memory.
            # 2GB keeps total VRAM ~9.5GB (safe margin) with fast algorithms.
            "trt_max_workspace_size": self._workspace_gb * 1024 * 1024 * 1024,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_base),
        }
        if self._qdq:
            # 显式 QDQ 图（FP8 实验）：引擎与 FP16 分开缓存。
            # FP8 QDQ 必须 strongly-typed 构建模式——默认隐式模式只认 int8 QDQ，
            # 会把 FP8 Q/DQ 整个丢弃（实测输出与 FP16 逐位相同）。
            qdq_dir = cache_base / "qdq" / self._label
            qdq_dir.mkdir(parents=True, exist_ok=True)
            trt_opts["trt_engine_cache_path"] = str(qdq_dir)
            if os.environ.get("SHARP3D_TRT_STRONG") == "1":
                trt_opts["trt_building_mode"] = "strongly_typed"
        # CUDA Graph is opt-in only (SHARP3D_TRT_CUDAGRAPH=1): it needs
        # persistent fixed I/O buffers (~475MB held outside the caching
        # allocator's reuse pool), which pushed the 12GB-VRAM VR pipeline
        # into shared-memory paging, and measured no encoder speedup
        # (213ms → 223ms). Left as an escape hatch for bigger GPUs.
        if os.environ.get("SHARP3D_TRT_CUDAGRAPH") == "1":
            trt_opts["trt_cuda_graph_enable"] = True

        providers = [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]

        try:
            session = ort.InferenceSession(
                str(self._onnx_path), sess_options=so, providers=providers
            )
        except Exception as e:
            if not trt_opts.get("trt_cuda_graph_enable"):
                raise
            # Older ORT builds reject trt_cuda_graph_enable — retry without.
            logger.warning("TRT CUDA Graph unsupported (%s), retrying without", e)
            trt_opts.pop("trt_cuda_graph_enable", None)
            providers[0] = ("TensorrtExecutionProvider", trt_opts)
            session = ort.InferenceSession(
                str(self._onnx_path), sess_options=so, providers=providers
            )
        active = session.get_providers()

        if "TensorrtExecutionProvider" in active:
            logger.info("Using TensorRT FP16 for %s", self._label)
        elif "CUDAExecutionProvider" in active:
            logger.info("TensorRT unavailable, using CUDA EP for %s", self._label)
        else:
            logger.warning("GPU EPs unavailable for %s, using CPU (slow!)", self._label)

        return session

    def _resolve_shape(self, template, batch):
        """Replace dynamic dimension placeholder with actual batch size."""
        return tuple(batch if d == -1 else d for d in template)

    def _forward_iobinding(self, x: torch.Tensor):
        """Zero-copy IO Binding path: GPU tensor → ORT → GPU tensors.

        Outputs are allocated per call through PyTorch's caching allocator so
        the blocks stay in the shared reuse pool between calls — holding
        fixed buffers instead permanently pins ~475MB and starved the VR
        pipeline into shared-memory paging on 12GB GPUs.
        """
        batch = x.shape[0]
        dev_id = self.device.index if self.device.index is not None else 0
        # Ensure FP32: after predictor.half(), patches may arrive as FP16,
        # but the ONNX model and IO Binding expect FP32 input.
        x_contig = x.detach().float().contiguous()

        # Pre-allocate output tensors on CUDA based on session metadata
        out_tensors = []
        for template in self._output_shape_templates:
            shape = self._resolve_shape(template, batch)
            t = torch.empty(shape, dtype=torch.float32, device=self.device)
            out_tensors.append(t)

        # torch.compile launches the patch-producing kernels asynchronously on
        # PyTorch's stream, but ORT runs inference on its own separate stream.
        # Without this sync, ORT can read patches that are not yet fully
        # produced → corrupted ViT features → intermittent blurry frames.
        # Only the *current* stream needs draining: a device-wide
        # torch.cuda.synchronize() would also wait for the video pipeline's
        # side-stream H2D prefetch and defeat the double-buffering.
        torch.cuda.current_stream().synchronize()

        # Create IO binding
        binding = self._session.io_binding()

        # Bind input (GPU tensor directly, no CPU transfer)
        binding.bind_input(
            name=self._input_name,
            device_type='cuda',
            device_id=dev_id,
            element_type=np.float32,
            shape=tuple(x_contig.shape),
            buffer_ptr=x_contig.data_ptr(),
        )

        # Bind outputs to pre-allocated CUDA tensors
        for name, t in zip(self._output_names, out_tensors):
            binding.bind_output(
                name=name,
                device_type='cuda',
                device_id=dev_id,
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
        # Must mirror the FP32 cast in _forward_iobinding (see the comment
        # there): after predictor.half() the patches arrive as float16, and the
        # ONNX session declares float32 inputs. Without this the numpy path —
        # which is exactly where we land when IO Binding fails — raises a dtype
        # mismatch instead of producing a result.
        x_np = x.detach().float().cpu().numpy()
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
        # Stream-ordering sync lives inside _forward_iobinding (after the
        # input staging copy); the numpy fallback path syncs via .cpu().
        _t0 = time.perf_counter() if profiling.ENABLED else 0.0
        if not self._iobinding_failed:
            try:
                out = self._forward_iobinding(x)
                self._iobinding_fail_streak = 0   # recovered — clear the streak
                if profiling.ENABLED:
                    profiling.add_ort(self._label,
                                      (time.perf_counter() - _t0) * 1000.0)
                return out
            except Exception as e:
                self._iobinding_fail_streak += 1
                if self._iobinding_fail_streak >= self._IOBINDING_MAX_FAILS:
                    # Structural: latch to numpy for good, but say so loudly —
                    # this is a multi-x slowdown, not a cosmetic warning.
                    self._iobinding_failed = True
                    logger.error(
                        "IO Binding 连续失败 %d 次，已永久回退 numpy 路径"
                        "（性能显著下降，GPU 零拷贝加速失效）: %s",
                        self._iobinding_fail_streak, e)
                else:
                    logger.warning("IO Binding failed (%d/%d), using numpy for "
                                   "this frame: %s",
                                   self._iobinding_fail_streak,
                                   self._IOBINDING_MAX_FAILS, e)
        else:
            logger.debug("IO Binding disabled earlier; using numpy path")
        out = self._forward_numpy(x)
        if profiling.ENABLED:
            profiling.add_ort(self._label, (time.perf_counter() - _t0) * 1000.0)
        return out


# ── Factory functions ────────────────────────────────────────────────

# ── Full-pipeline ORT orchestrator (3 sessions, zero CPU roundtrip) ──
#
#   [pyramid+split (torch glue, ~2ms)] → patch ViT session → image ViT
#   session → SPN-tail session (merge/upsample + decoder + composer, FP16)
#
# Why not one big graph: INT8 QDQ calibration of the full graph OOMs a 12GB
# GPU (24 blocks × ~1.6GB of QDQ transients co-resident with everything
# else). Calibrating the standalone ViT graphs yields identical scales —
# the ViT activation ranges depend only on the ViT inputs.

class ORTRestSession:
    """IO-bound multi-input session. fp32=True → CUDA EP only (no TRT):
    used for the decoder+composer half whose GroupNorm-family ops degrade
    under TRT fp16 (autocast keeps them fp32; TRT fp16 cannot)."""

    def __init__(self, onnx_path: Path, trt_cache_dir: Path,
                 device: torch.device, label: str, workspace_gb: int = 2,
                 fp32: bool = False):
        _ensure_cudnn_path()
        import onnxruntime as ort
        self.device = device
        self._label = label
        self.using_trt = False
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 2
        if fp32:
            providers = [("CUDAExecutionProvider", {"device_id": 0}),
                         "CPUExecutionProvider"]
            logger.info("FP32 CUDA 会话就绪: %s", label)
        else:
            engine_dir = _ascii_safe_trt_dir(trt_cache_dir) / "tail" / label
            if workspace_gb != 2:
                # Same rationale as ORTEncoder: key the engine cache on the
                # build-affecting workspace size; default keeps the historical
                # path so cached engines (label "35"/"21") stay valid.
                engine_dir = engine_dir.parent / f"{label}_ws{workspace_gb}"
            engine_dir.mkdir(parents=True, exist_ok=True)
            trt_opts = {
                "trt_fp16_enable": True,
                "trt_max_workspace_size": workspace_gb * 1024**3,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(engine_dir),
                # torch.autocast keeps norm layers in fp32; force the same
                # here (full-fp16 engines measurably degrade output quality).
                "trt_layer_norm_fp32_fallback": True,
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         ("CUDAExecutionProvider", {"device_id": 0}),
                         "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(onnx_path), sess_options=so,
                                             providers=providers)
        self.using_trt = ("TensorrtExecutionProvider"
                          in self._session.get_providers())
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._output_shapes = [tuple(-1 if isinstance(d, str) else int(d)
                                     for d in o.shape)
                               for o in self._session.get_outputs()]
        if self.using_trt:
            logger.info("TRT FP16 尾部引擎就绪: %s", label)
        else:
            logger.warning("TRT EP 不可用，SPN 尾部回退 CUDA EP")

    def run(self, tensors):
        dev_id = self.device.index if self.device.index is not None else 0
        feats = [t.detach().float().contiguous() for t in tensors]
        outs = [torch.empty(s, dtype=torch.float32, device=self.device)
                for s in self._output_shapes]
        torch.cuda.current_stream().synchronize()
        binding = self._session.io_binding()
        for name, t in zip(self._input_names, feats):
            binding.bind_input(name=name, device_type="cuda", device_id=dev_id,
                               element_type=np.float32, shape=tuple(t.shape),
                               buffer_ptr=t.data_ptr())
        for name, t in zip(self._output_names, outs):
            binding.bind_output(name=name, device_type="cuda", device_id=dev_id,
                                element_type=np.float32, shape=tuple(t.shape),
                                buffer_ptr=t.data_ptr())
        self._session.run_with_iobinding(binding)
        return outs


class ORTFullPredictor(nn.Module):
    """3-session orchestrator replacing the whole RGBGaussianPredictor.

    forward(): pyramid+split in plain torch ops (~2ms), patch/image ViT
    sessions (INT8 QDQ when available), SPN-tail session (FP16). All hops
    are IO-bound GPU→GPU; the Gaussians3D NamedTuple is rebuilt from the 5
    outputs, matching the torch path's output format exactly.
    """

    _OUT_FIELDS = ("mean_vectors", "singular_values", "quaternions",
                   "colors", "opacities")

    def __init__(self, patch: "ORTEncoder", image: "ORTEncoder",
                 front: ORTRestSession, rest: ORTRestSession,
                 device: torch.device, n_patches: int, normalizer=None,
                 tail_fp16: bool = False):
        super().__init__()
        from sharp.utils.gaussians import Gaussians3D
        self._Gaussians3D = Gaussians3D
        self.device = device
        self._patch = patch
        self._image = image
        self._front = front
        self._rest = rest
        self._n_patches = n_patches
        # Production applies AffineRangeNormalizer before the pyramid (the
        # ViTs see [-1,1] imagery) but passes the RAW image to init_model.
        self._normalizer = normalizer
        # 尾部 fp16 TRT 模式（SHARP3D_FULL_TRT_TAILFP16=1）：用整段尾部引擎
        # （实测 35.1dB），替代 front+fp32-rest 拆分（实测 47.8dB 但 front
        # 引擎构建尚有 bug）。
        self._tail_fp16 = tail_fp16
        # 加速判定只看 ViT（尾部按所选模式各自判定）
        self.using_trt = bool(patch.using_trt and image.using_trt
                              and rest.using_trt)

    @torch.compiler.disable(recursive=False)
    def forward(self, img, disparity_factor):
        import torch.nn.functional as F
        from sharp.models.encoders.spn_encoder import split

        _t0 = time.perf_counter() if profiling.ENABLED else 0.0
        img = img.detach().float()
        # Normalized imagery for the ViT path (production feeds the
        # normalizer output into the pyramid); RAW image goes to the tail
        # (init_model) — same as RGBGaussianPredictor.forward.
        if self._normalizer is not None:
            xn = self._normalizer(img)
        else:
            xn = img
        # Pyramid + sliding-window patches — identical math to
        # SlidingPyramidNetwork._create_pyramid + split. Overlap ratios and
        # patch counts follow the perf mode (35 = 5×5+3×3+1×1 @ .25/.5,
        # 21 = 4×4+2×2+1×1 @ 0/0).
        x1 = F.interpolate(xn, scale_factor=0.5, mode="bilinear",
                           align_corners=False)
        x2 = F.interpolate(xn, scale_factor=0.25, mode="bilinear",
                           align_corners=False)
        if self._n_patches == 35:
            patches = torch.cat((
                split(xn, overlap_ratio=0.25, patch_size=384),
                split(x1, overlap_ratio=0.5, patch_size=384),
                x2), dim=0)
        elif self._n_patches == 21:
            # speed mode: 21 = 4×4 + 2×2 + 1×1, no overlap
            patches = torch.cat((
                split(xn, overlap_ratio=0.0, patch_size=384),
                split(x1, overlap_ratio=0.0, patch_size=384),
                x2), dim=0)
        else:
            # Fail loudly: a silent wrong-branch would feed the wrong number of
            # patches to graphs built for a different layout, producing either a
            # shape error deep in the tail or plausible-looking wrong geometry.
            raise ValueError(
                f"不支持的 n_patches={self._n_patches}（仅 35 / 21）")

        pe_feat, pe_ints = self._patch(patches)
        ie_feat, _ = self._image(x2)
        ids = self._patch.intermediate_features_ids
        if self._tail_fp16:
            # 单段尾部引擎（TRT fp16）：整段 SPN+解码器+composer
            outs = self._rest.run([img, disparity_factor.detach().float().reshape(-1),
                                   pe_feat, pe_ints[ids[0]], pe_ints[ids[1]],
                                   ie_feat])
        else:
            encs = self._front.run([pe_feat, pe_ints[ids[0]], pe_ints[ids[1]],
                                    ie_feat])
            outs = self._rest.run([img,
                                   disparity_factor.detach().float().reshape(-1),
                                   *encs])
        if profiling.ENABLED:
            profiling.add_ort("full_3session",
                              (time.perf_counter() - _t0) * 1000.0)
        return self._Gaussians3D(*outs)


def create_ort_full_predictor(
    predictor,
    device: torch.device,
    n_patches: int,
    onnx_dir: Path = _ONNX_DIR,
    trt_cache_dir: Path = _TRT_CACHE_DIR,
) -> "ORTFullPredictor | None":
    """Assemble the 3-session pipeline; None → caller falls back to torch.

    INT8 QDQ support was removed (2026-09-10): the QDQ graph degrades image
    quality to 25dB and TRT EP never built its engines anyway — see
    tests/quantize_vit_int8.py for the (dead-end) tooling.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError as e:
        _log_ort_error(f"onnxruntime import failed: {e}")
        return None

    rest_onnx = onnx_dir / f"tail_rest_{n_patches}.onnx"
    front_onnx = onnx_dir / f"spn_front_{n_patches}.onnx"
    if not rest_onnx.exists() or not front_onnx.exists():
        # fp16 权重也能导（TRT front 按 fp16 走）；rest2 是 fp32 CUDA 会话。
        try:
            if any(p.dtype == torch.float16 for p in predictor.parameters()):
                predictor.float()   # export in fp32; caller re-halves on fallback
            from .full_export import export_spn_split
            export_spn_split(predictor, device, n_patches, onnx_dir)
        except Exception:
            import traceback
            _log_ort_error(f"SPN 拆分导出失败:\n{traceback.format_exc()}")
            return None

    patch_onnx = onnx_dir / "patch_encoder.onnx"
    image_onnx = onnx_dir / "image_encoder.onnx"
    patch_label = "patch_encoder"
    qdq = False
    if os.environ.get("SHARP3D_PATCH_FP8") == "1":
        p8 = onnx_dir / "patch_encoder_fp8.onnx"
        if p8.exists():
            patch_onnx, patch_label, qdq = p8, "patch_encoder_fp8", True
            logger.info("使用 FP8 QDQ patch encoder（实验）")
        else:
            logger.info("FP8 patch encoder 不存在，回退 FP16 "
                        "（生成: python tests/fp8_surgery.py）")

    try:
        patch = ORTEncoder(patch_onnx, trt_cache_dir, device,
                           label=patch_label, qdq=qdq)
        image = ORTEncoder(image_onnx, trt_cache_dir, device,
                           label="image_encoder")
        tail_fp16 = os.environ.get("SHARP3D_FULL_TRT_TAILFP16") == "1"
        if tail_fp16:
            # 单段尾部（spn_tail）TRT fp16 —— 质量对照样本用配置。
            # label 必须沿用 "35"：同一 ONNX + 相同构建选项，直接复用已缓存
            # 的引擎（用新 label 会触发 ~8 分钟的引擎重建）。
            tail_onnx = onnx_dir / f"spn_tail_{n_patches}.onnx"
            if not tail_onnx.exists():
                logger.info("spn_tail ONNX 不存在，无法用 TAILFP16 模式")
                return None
            front = ORTRestSession(tail_onnx, trt_cache_dir, device,
                                   label=str(n_patches))
            rest = front   # forward 分支会直接用 rest，不经过 front
        else:
            front = ORTRestSession(front_onnx, trt_cache_dir, device,
                                   label=str(n_patches))
            rest = ORTRestSession(rest_onnx, trt_cache_dir, device,
                                  label=str(n_patches), fp32=True)
    except Exception:
        import traceback
        _log_ort_error(f"整模型 TRT 会话创建失败:\n{traceback.format_exc()}")
        return None
    # normalizer lives on the monodepth predictor (traced nowhere — the ViTs
    # are graph inputs now), applied in the orchestrator forward.
    normalizer = predictor.monodepth_model.monodepth_predictor.normalizer
    return ORTFullPredictor(patch, image, front, rest, device, n_patches,
                            normalizer=normalizer, tail_fp16=tail_fp16)


def create_ort_patch_encoder(
    predictor,
    device: torch.device,
    onnx_dir: Path = _ONNX_DIR,
    trt_cache_dir: Path = _TRT_CACHE_DIR,
) -> ORTEncoder | None:
    """Create an ORT-accelerated patch_encoder, exporting ONNX if needed."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError as e:
        _log_ort_error(f"onnxruntime import failed: {e}")
        return None

    # Frozen: use bundled ONNX if available
    import sys as _sys
    if getattr(_sys, "frozen", False):
        bundled = Path(_sys._MEIPASS) / "models" / "patch_encoder.onnx"
        if bundled.exists():
            onnx_path = bundled
        else:
            onnx_path = onnx_dir / "patch_encoder.onnx"
    else:
        onnx_path = onnx_dir / "patch_encoder.onnx"

    if not onnx_path.exists():
        try:
            export_patch_encoder(predictor, onnx_path, device)
        except Exception as e:
            import traceback
            _log_ort_error(f"ONNX export failed:\n{traceback.format_exc()}")
            return None

    try:
        return ORTEncoder(onnx_path, trt_cache_dir, device,
                          label="patch_encoder")
    except Exception as e:
        import traceback
        _log_ort_error(f"ORT session creation failed:\n{traceback.format_exc()}")
        return None


def create_ort_image_encoder(
    predictor,
    device: torch.device,
    onnx_dir: Path = _ONNX_DIR,
    trt_cache_dir: Path = _TRT_CACHE_DIR,
) -> ORTEncoder | None:
    """Create an ORT-accelerated image_encoder, exporting ONNX if needed."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None

    # Frozen: use bundled ONNX if available
    import sys as _sys
    if getattr(_sys, "frozen", False):
        bundled = Path(_sys._MEIPASS) / "models" / "image_encoder.onnx"
        if bundled.exists():
            onnx_path = bundled
        else:
            onnx_path = onnx_dir / "image_encoder.onnx"
    else:
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
                          label="image_encoder")
    except Exception as e:
        logger.warning("Image encoder ORT session creation failed: %s", e)
        return None
