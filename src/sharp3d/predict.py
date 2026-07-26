"""SHARP model loading and compiled inference (single authority for model setup).

Consolidates all model loading optimizations:
    - FP16 checkpoint caching (mmap, half disk I/O)
    - ORT TensorRT acceleration (patch_encoder + image_encoder)
    - channels_last memory format for Conv2d
    - torch.compile (max-autotune, Windows-safe config)
    - TRT engine pre-build + gsplat kernel warmup
    - INT8 mode for speed-priority

BUG#7 RESOLVED: PyTorch 2.13 fixes the "Python int too large to convert to C long"
    error on Windows. mode="max-autotune" now works correctly.

BUG#8 FIX: CUDA non-default streams are incompatible with torch.compile
    on Windows (Triton limitation → OverflowError). No dual-stream pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
from sharp.models import PredictorParams, create_predictor

logger = logging.getLogger(__name__)

MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# Type alias for progress callback: (stage_description, percent_0_100)
ProgressCB = Callable[[str, int], None] | None


class SharpPredictor:
    """SHARP predictor with full optimization stack.

    Usage:
        sp = SharpPredictor(device, perf_mode="quality", progress_cb=my_cb)
        g_ndc = sp.predict(img_resized, disparity_factor)
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        perf_mode: str = "quality",
        cache_dir: Path | None = None,
        progress_cb: ProgressCB = None,
    ):
        """
        Args:
            device: CUDA device.
            perf_mode: "quality" (35 patches) or "speed" (21 patches + INT8).
            cache_dir: Directory for FP16 weights + TRT/inductor/triton caches.
                       Defaults to <project_root>/.cache.
            progress_cb: Optional (stage: str, percent: int) callback for UI.
        """
        self.device = device
        self.perf_mode = perf_mode
        self._progress = progress_cb or (lambda s, p: None)
        # Track acceleration methods: list of (name, enabled)
        self.accel_status: list[tuple[str, bool]] = []

        if cache_dir is None:
            import sys as _sys, os as _os
            if getattr(_sys, "frozen", False):
                cache_dir = Path(_os.environ.get("LOCALAPPDATA", "~")) / "sharp3d" / ".cache"
            else:
                cache_dir = Path(__file__).resolve().parents[2] / ".cache"
        self._cache_dir = cache_dir

        # ── Load weights ─────────────────────────────────────────────────
        self._progress("加载模型权重", 8)
        params = PredictorParams()
        if perf_mode == "speed":
            params.monodepth.use_patch_overlap = False

        # Frozen: check bundled weights first
        import sys as _sys
        fp16_ckpt = cache_dir / "sharp_fp16.pt"
        if getattr(_sys, "frozen", False):
            bundled = Path(_sys._MEIPASS) / "models" / "sharp_fp16.pt"
            if bundled.exists():
                fp16_ckpt = bundled
        if fp16_ckpt.exists():
            state_dict = torch.load(str(fp16_ckpt), map_location="cpu",
                                    mmap=True, weights_only=True)
            already_fp16 = True
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                MODEL_URL, progress=False, map_location="cpu")
            already_fp16 = False

        predictor = create_predictor(params)
        predictor.load_state_dict(state_dict)
        del state_dict
        predictor.eval().to(device)
        self.predictor = predictor

        # Save FP16 cache for next startup
        if not already_fp16:
            try:
                fp16_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.half() for k, v in predictor.state_dict().items()},
                           str(fp16_ckpt))
            except Exception:
                pass
        self.accel_status.append(("FP16 权重", True))

        # ── channels_last (lossless Conv2d speedup) ──────────────────────
        _cl_ok = False
        try:
            for mod in predictor.modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    mod.weight.data = mod.weight.data.to(
                        memory_format=torch.channels_last)
            _cl_ok = True
        except Exception:
            pass
        self.accel_status.append(("channels_last", _cl_ok))

        # ── ORT TensorRT acceleration ────────────────────────────────────
        use_int8 = (perf_mode == "speed")
        _ort_ok = False
        try:
            from .ort_engine import create_ort_patch_encoder, create_ort_image_encoder
            spn = predictor.monodepth_model.monodepth_predictor.encoder

            ort_enc = create_ort_patch_encoder(predictor, device, int8_enable=use_int8)
            if ort_enc is not None:
                spn.patch_encoder = ort_enc

            ort_img = create_ort_image_encoder(predictor, device, int8_enable=use_int8)
            if ort_img is not None:
                spn.image_encoder = ort_img

            _ort_ok = (ort_enc is not None or ort_img is not None)
        except Exception as e:
            import traceback, os as _os2
            logger.warning("ORT TensorRT 引擎构建失败: %s", e)
            try:
                Path(_os2.environ.get("LOCALAPPDATA", ".")).joinpath(
                    "sharp3d", "ort_error.log").write_text(
                    traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
        _ort_label = "ORT TensorRT" + (" INT8" if use_int8 else "")
        self.accel_status.append((_ort_label, _ort_ok))

        # ── FP16 conversion (after ORT export which needs FP32) ──────────
        if not already_fp16:
            predictor.half()

        # ── Detect cache state ───────────────────────────────────────────
        trt_cached = (cache_dir / "trt_v3").exists() and any(
            (cache_dir / "trt_v3").glob("*.engine"))
        inductor_cached = (cache_dir / "inductor").exists() and any(
            (cache_dir / "inductor").rglob("*.py"))
        triton_cached = (cache_dir / "triton").exists() and (
            any((cache_dir / "triton").rglob("*.so")) or
            any((cache_dir / "triton").rglob("*.ptx")))

        # ── Pre-build TRT engines ────────────────────────────────────────
        try:
            spn = predictor.monodepth_model.monodepth_predictor.encoder
            if hasattr(spn.patch_encoder, '_session'):
                n_patches = 35 if params.monodepth.use_patch_overlap else 21
                self._progress("TensorRT 引擎", 35)
                dummy = torch.zeros(n_patches, 3, 384, 384, device=device)
                with torch.no_grad():
                    spn.patch_encoder(dummy)
                torch.cuda.synchronize()
                del dummy
            if hasattr(spn.image_encoder, '_session'):
                dummy = torch.zeros(1, 3, 384, 384, device=device)
                with torch.no_grad():
                    spn.image_encoder(dummy)
                torch.cuda.synchronize()
                del dummy
        except Exception:
            pass

        # ── torch.compile (triton JIT 编译，无需 MSVC；失败则回退 eager) ──
        import os as _os
        if _os.environ.get("SHARP3D_NO_COMPILE"):
            self._progress("跳过编译（打包模式）", 55)
            self._compiled = predictor
        else:
            try:
                self._progress("编译预测器", 55)
                torch._dynamo.config.capture_scalar_outputs = True
                torch._inductor.config.triton.cudagraphs = False
                torch._inductor.config.compile_threads = 1
                torch._inductor.config.coordinate_descent_tuning = False
                self._compiled = torch.compile(predictor, mode="max-autotune", dynamic=False)
            except Exception as e:
                self._progress("跳过编译（回退 eager）", 55)
                import traceback
                _err = traceback.format_exc()
                logger.warning("torch.compile 失败，回退 eager 模式: %s", e)
                try:
                    Path(_os.environ.get("LOCALAPPDATA", ".")).joinpath(
                        "sharp3d", "compile_error.log").write_text(_err, encoding="utf-8")
                except Exception:
                    pass
                self._compiled = predictor
        _is_compiled = hasattr(self._compiled, "_orig_mod")
        _mode = "torch.compile 已启用" if _is_compiled else "eager 模式（未编译）"
        logger.info("torch.compile 状态: %s", _mode)
        self.accel_status.append(("torch.compile", _is_compiled))
        self._progress(_mode, 60)

        # ── Warmup inference ─────────────────────────────────────────────
        from .unproject import INTERNAL_SHAPE
        dummy_img = torch.zeros(1, 3, *INTERNAL_SHAPE, device=device)
        dummy_df = torch.tensor([1.0], device=device, dtype=torch.float32)
        self._progress("预热推理", 75)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g_ndc = self._compiled(dummy_img, dummy_df)
        torch.cuda.synchronize()

        # ── gsplat render kernel warmup ──────────────────────────────────
        if not triton_cached:
            self._progress("编译渲染内核", 90)
            try:
                from .unproject import fast_unproject
                from .render import render_sbs
                f = INTERNAL_SHAPE[1] * 1.2
                ir = torch.tensor([
                    [f, 0, (INTERNAL_SHAPE[1] - 1) / 2.0, 0],
                    [0, f, (INTERNAL_SHAPE[0] - 1) / 2.0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ], dtype=torch.float32, device=device)
                g = fast_unproject(g_ndc, torch.eye(4, device=device), ir,
                                   INTERNAL_SHAPE, decompose_method="analytical")
                render_sbs(g, f, INTERNAL_SHAPE[1], INTERNAL_SHAPE[0],
                           ipd=0.063, render_width=320)
                torch.cuda.synchronize()
                del g, ir
            except Exception:
                pass

        del dummy_img, dummy_df, g_ndc
        torch.cuda.empty_cache()
        self._progress("就绪", 100)

    @torch.no_grad()
    def predict(self, img_resized: torch.Tensor,
                disparity_factor: torch.Tensor):
        """Run SHARP prediction (compiled + FP16 autocast).

        Args:
            img_resized: (1, 3, 1536, 1536) float tensor.
            disparity_factor: (1,) float32 tensor.

        Returns:
            Gaussians3D in NDC space.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            return self._compiled(img_resized, disparity_factor)

    def __call__(self, img_resized: torch.Tensor, disparity_factor: torch.Tensor):
        """Allow using SharpPredictor directly as predict_fn."""
        return self.predict(img_resized, disparity_factor)
