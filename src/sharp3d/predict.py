"""SHARP model loading and compiled inference (single authority for model setup).

Consolidates all model loading optimizations:
    - FP16 checkpoint caching (mmap, half disk I/O)
    - ORT TensorRT acceleration (patch_encoder + image_encoder)
    - channels_last memory format for Conv2d
    - torch.compile (max-autotune, Windows-safe config)
    - TRT engine pre-build + gsplat kernel warmup

BUG#7 RESOLVED: PyTorch 2.13 fixes the "Python int too large to convert to C long"
    error on Windows. mode="max-autotune" now works correctly.

BUG#8 FIX: CUDA non-default streams are incompatible with torch.compile
    on Windows (Triton limitation → OverflowError). No dual-stream pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import torch
from sharp.models import PredictorParams, create_predictor

logger = logging.getLogger(__name__)

MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# Type alias for progress callback: (stage_description, percent_0_100)
ProgressCB = Callable[[str, int], None] | None


def _write_startup_log(cache_dir: Path, device, perf_mode: str) -> None:
    """Record the facts that decide whether startup will be fast or slow.

    Frozen builds run with console=False, so stdout/stderr go nowhere and a
    hang is indistinguishable from slow work. This file is the only way to
    tell which machine hit which path — it records the GPU (TensorRT engines
    are NOT portable between architectures), whether the cached artifacts the
    slow steps depend on actually exist, and whether the cache dir is writable.
    """
    try:
        import os
        import sys
        import time

        log = Path(os.environ.get("LOCALAPPDATA", ".")) / "sharp3d" / "startup.log"
        log.parent.mkdir(parents=True, exist_ok=True)

        gpu = "?"
        try:
            import torch
            if device.type == "cuda":
                gpu = f"{torch.cuda.get_device_name(device)} (sm{torch.cuda.get_device_capability(device)[0]}{torch.cuda.get_device_capability(device)[1]})"
        except Exception as exc:
            gpu = f"? ({exc})"

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            probe = cache_dir / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            writable = "yes"
        except Exception as exc:
            writable = f"NO ({exc})"

        def _have(rel: str) -> str:
            p = cache_dir / rel
            if not p.exists():
                return "missing"
            sz = p.stat().st_size / 2**20 if p.is_file() else 0
            return f"{sz:.0f}MB" if p.is_file() else "dir"

        engines = sorted((cache_dir / "trt_v3").glob("*.engine")) if (cache_dir / "trt_v3").is_dir() else []
        # The TRT EP encodes the target SM in the engine filename; an engine
        # built for one GPU architecture cannot be loaded on another.
        sm_tags = sorted({e.name.split("_")[-1].replace(".engine", "") for e in engines})

        bundled = "n/a (source checkout)"
        if getattr(sys, "frozen", False):
            b = Path(sys._MEIPASS) / "models" / "sharp_fp16.pt"
            bundled = f"{b.stat().st_size / 2**20:.0f}MB" if b.exists() else "MISSING -> will download!"

        from . import __version__

        with open(log, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                f"version       : {__version__}\n"
                f"frozen        : {getattr(sys, 'frozen', False)}\n"
                f"gpu           : {gpu}\n"
                f"perf_mode     : {perf_mode}\n"
                f"cache_dir     : {cache_dir}\n"
                f"cache_writable: {writable}\n"
                f"bundled_ckpt  : {bundled}\n"
                f"fp16_ckpt     : {_have('sharp_fp16.pt')}\n"
                f"onnx patch    : {_have('onnx/patch_encoder.onnx')}\n"
                f"onnx image    : {_have('onnx/image_encoder.onnx')}\n"
                f"trt engines   : {len(engines)} (built for: {', '.join(sm_tags) or 'none'})\n"
                f"inductor      : {_have('inductor')}\n"
                f"triton        : {_have('triton')}\n"
            )
    except Exception:
        pass


class SharpPredictor:
    """SHARP predictor with full optimization stack.

    Usage:
        sp = SharpPredictor(device, perf_mode="quality", progress_cb=my_cb)
        g_ndc = sp.predict(img_resized, disparity_factor)
    """

    @property
    def _ac_dtype(self):
        """Autocast dtype matching the weight precision (FP16 or FP32)."""
        return torch.float16 if self._fp16 else torch.float32

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        perf_mode: str = "quality",
        cache_dir: Path | None = None,
        progress_cb: ProgressCB = None,
        fp16: bool = True,
    ):
        """
        Args:
            device: CUDA device.
            perf_mode: "quality" (35 patches) or "speed" (21 patches).
            cache_dir: Directory for FP16 weights + TRT/inductor/triton caches.
                       Defaults to <project_root>/.cache.
            progress_cb: Optional (stage: str, percent: int) callback for UI.
            fp16: Run the predictor weights and autocast in FP16. False keeps
                  FP32 end to end (slower, ~2x VRAM, but no FP16 rounding).

                  INT8 was removed (2026-09-10): ORT TRT EP ignores the implicit
                  calibration table for this ViT (all layers fall back to FP16 —
                  measured pixel-identical output), and explicit QDQ static
                  quantization degrades image quality to 25dB. Dead end without
                  quantization-aware training.
        """
        self.device = device
        self.perf_mode = perf_mode
        self._fp16 = fp16
        # Set on first-forward dynamo/triton failure (see predict()); compile
        # is lazy so __init__'s try/except cannot cover that stage.
        self._eager_fallback = False
        self._progress = progress_cb or (lambda s, p: None)
        # Track acceleration methods: list of (name, enabled)
        self.accel_status: list[tuple[str, bool]] = []

        if cache_dir is None:
            from . import resolve_cache_dir
            cache_dir = resolve_cache_dir()
        self._cache_dir = cache_dir
        _write_startup_log(cache_dir, device, perf_mode)

        # NOTE: torch.backends.cudnn.benchmark=True was measured (2026-09-09)
        # to be a wash for the SPN convs (predict 412.6 -> 420.4ms, conv
        # kernels 124 -> 105ms but total unchanged) and it adds per-shape
        # autotune cost at startup. Fixed conv shapes here are NOT
        # algo-selection-limited; don't re-enable without new evidence.

        # ── Load weights ─────────────────────────────────────────────────
        self._progress("加载模型权重", 8)
        params = PredictorParams()
        if perf_mode == "speed":
            params.monodepth.use_patch_overlap = False

        # Frozen: check bundled weights first
        import sys as _sys
        fp16_ckpt = cache_dir / "sharp_fp16.pt"
        fp32_ckpt = cache_dir / "sharp_fp32.pt"
        if getattr(_sys, "frozen", False):
            bundled = Path(_sys._MEIPASS) / "models" / "sharp_fp16.pt"
            if bundled.exists():
                fp16_ckpt = bundled
        if not fp16 and fp32_ckpt.exists():
            # FP32 pipeline: load the original Apple FP32 checkpoint — the
            # bundled sharp_fp16.pt has already lost precision at save time,
            # so casting its values up would NOT give a true FP32 model.
            self._progress("加载本地 FP32 权重 (sharp_fp32.pt)", 10)
            state_dict = torch.load(str(fp32_ckpt), map_location="cpu",
                                    mmap=True, weights_only=True)
            already_fp16 = False
        elif fp16_ckpt.exists() and fp16:
            self._progress(f"加载本地权重 ({fp16_ckpt.name})", 10)
            state_dict = torch.load(str(fp16_ckpt), map_location="cpu",
                                    mmap=True, weights_only=True)
            already_fp16 = True
        else:
            # No bundled/cache weights: 2.4GB over the network with no
            # progress reporting. Say so plainly instead of appearing frozen.
            self._progress("下载模型权重 (首次需联网，约2.4GB)", 9)
            state_dict = torch.hub.load_state_dict_from_url(
                MODEL_URL, progress=False, map_location="cpu")
            already_fp16 = False

        # torch.load(mmap=True) is lazy: the 1.4GB page-in actually happens in
        # load_state_dict below. Report each step so a slow disk cannot be
        # mistaken for a hang.
        self._progress("构建网络结构", 15)
        predictor = create_predictor(params)
        self._progress("载入权重", 20)

        # When the ONNX exports are already on disk the predictor never needs
        # FP32 weights: converting the freshly built model to FP16 *before*
        # load_state_dict means the mmap'd FP16 checkpoint is copied straight
        # into FP16 params. Otherwise load_state_dict would first materialize
        # a full FP32 model (~2.8GB) whose only purpose is to be halved —
        # doubling both the RAM peak and the weights' H2D transfer, which
        # matters a lot on 8GB-GPU machines.
        import os as _os
        from .ort_engine import onnx_models_cached
        direct_fp16 = bool(
            already_fp16 and fp16 and onnx_models_cached()
            and _os.environ.get("SHARP3D_NO_DIRECT_FP16") != "1")
        if direct_fp16:
            predictor.half()

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
        self.accel_status.append((f"{'FP16' if fp16 else 'FP32'} 权重", True))

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
        # Building the TRT engines is by far the slowest startup step and it
        # is unavoidable whenever there is no engine matching *this* GPU yet
        # (engines are architecture-specific: one built on sm_120 cannot load
        # on sm_89). It used to happen with no progress update at all, so the
        # UI sat at ~20% for minutes looking frozen.
        _trt_dir = cache_dir / "trt_v3"

        # ── 整模型 TRT（实验性，默认关闭）────────────────────────────────
        # 单图 TRT fp16 与拆分式（SPN front fp16 + 解码器 fp32 CUDA）均已
        # 实现，但质量未达验收线：
        #   - 单图 fp16 引擎 35.4dB（GroupNorm 系算子被 fp16 化，autocast
        #     在 torch 里会留 fp32，TRT 无此策略；lnf32 回退只覆盖 LayerNorm）
        #   - INT8 QDQ 静态量化 25dB（MinMax/Percentile 都不够，需 QAT 级方案）
        # 默认走经过验证的旧路径（fp16 ViT TRT + torch.compile SPN，
        # 47.8dB）。实验路径：set SHARP3D_FULL_TRT=1。
        self._full_trt = None
        _n_patches = 35 if params.monodepth.use_patch_overlap else 21
        if fp16 and os.environ.get("SHARP3D_FULL_TRT") == "1":
            _have_full_engine = (_trt_dir / "full" / str(_n_patches)).is_dir() \
                and any((_trt_dir / "full" / str(_n_patches)).glob("*.engine"))
            self._progress(
                "整模型 TRT 引擎（就绪）" if _have_full_engine
                else "整模型 TRT 引擎（首次构建，约 10-30 分钟）", 22)
            try:
                from .ort_engine import create_ort_full_predictor
                full = create_ort_full_predictor(predictor, device, _n_patches)
                if full is not None and full.using_trt:
                    self._full_trt = full
            except Exception as e:
                logger.warning("整模型 TRT 初始化失败: %s", e)

        if self._full_trt is not None:
            # The session computes everything predict() needs — the torch
            # weights are dead weight on the GPU. Offload to CPU (saves
            # ~2.8GB VRAM in fp32 / 1.4GB in fp16).
            predictor.to("cpu")
            self._compiled = self._full_trt
            self.accel_status.append(("整模型 TRT FP16", True))
            self._progress("整模型 TRT 就绪（跳过 torch.compile）", 60)

        _ort_ok = False
        if not fp16:
            # FP32 pipeline: keep everything in torch FP32 — swapping in the
            # FP16 TRT encoder sessions would defeat the point of the mode.
            self.accel_status.append(("FP32 全精度 (纯 torch)", True))
            self._progress("FP32 模式（跳过 TensorRT）", 35)
        elif self._full_trt is None:
            _have_engine = _trt_dir.is_dir() and any(_trt_dir.glob("*.engine"))
            self._progress(
                "TensorRT 引擎（就绪）" if _have_engine
                else "TensorRT 引擎（首次构建，需数分钟）", 25)
            try:
                from .ort_engine import (create_ort_patch_encoder,
                                         create_ort_image_encoder, _TRT_CACHE_DIR)
                spn = predictor.monodepth_model.monodepth_predictor.encoder

                ort_enc = create_ort_patch_encoder(predictor, device)
                if ort_enc is not None:
                    spn.patch_encoder = ort_enc
                self._progress("TensorRT 引擎（2/2）", 30)

                ort_img = create_ort_image_encoder(predictor, device)
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
            self.accel_status.append(("ORT TensorRT", _ort_ok))

            # ── FP16 conversion (after ORT export which needs FP32) ──────
            # In FP16 mode the weights must end up FP16: when load_state_dict
            # fed FP16 values into FP32 params they were silently cast up, so
            # skipping half() here kept the model in FP32 (2× VRAM, ~1.4 GB
            # wasted). direct_fp16 already produced FP16 params — halving
            # again would be a no-op, and it must NOT run for fp16=False.
            if fp16 and not direct_fp16:
                predictor.half()

        # ── Detect cache state ───────────────────────────────────────────
        trt_cached = (cache_dir / "trt_v3").exists() and any(
            (cache_dir / "trt_v3").glob("*.engine"))
        inductor_cached = (cache_dir / "inductor").exists() and any(
            (cache_dir / "inductor").rglob("*.py"))
        triton_cached = (cache_dir / "triton").exists() and (
            any((cache_dir / "triton").rglob("*.so")) or
            any((cache_dir / "triton").rglob("*.ptx")))

        # ── Pre-build TRT engines (encoder-split path only) ─────────────
        if self._full_trt is None:
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
        else:
            self._progress("预热整模型 TRT 引擎", 35)

        # ── torch.compile (triton JIT 编译，无需 MSVC；失败则回退 eager) ──
        # 整模型 TRT 模式跳过：SPN 已在引擎里，编译 torch 权重没有意义。
        import os as _os
        if self._full_trt is not None:
            self.accel_status.append(("torch.compile", False))
        elif _os.environ.get("SHARP3D_NO_COMPILE"):
            self._progress("跳过编译（打包模式）", 55)
            self._compiled = predictor
        else:
            try:
                self._progress("编译预测器", 55)
                torch._dynamo.config.capture_scalar_outputs = True
                # CUDA graphs historically caused an OverflowError on Windows
                # when combined with the pipeline's non-default H2D stream
                # (BUG#8), so they stay off by default. SHARP3D_CUDAGRAPH=1
                # re-enables them as an opt-in experiment — the SPN decoder
                # launches many small kernels where graph replay saves real
                # launch overhead. Benchmark before shipping: if a run shows
                # the OverflowError again, leave this off.
                _cudagraphs = os.environ.get("SHARP3D_CUDAGRAPH") == "1"
                torch._inductor.config.triton.cudagraphs = _cudagraphs
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
        with torch.no_grad(), torch.autocast("cuda", dtype=self._ac_dtype):
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
        with torch.autocast("cuda", dtype=self._ac_dtype):
            if self._eager_fallback:
                return self.predictor(img_resized, disparity_factor)
            try:
                return self._compiled(img_resized, disparity_factor)
            except Exception as e:
                # torch.compile() is lazy: the dynamo/triton compile happens
                # HERE on the first forward, not in __init__ — an exception
                # at this point used to escape uncaught and hang model
                # loading (seen on clean machines: UnicodeDecodeError inside
                # dynamo). Degrade to eager permanently and log the full
                # traceback for diagnosis.
                import traceback
                self._eager_fallback = True
                logger.warning("torch.compile 首帧失败，回退 eager 模式: %s", e)
                try:
                    Path(os.environ.get("LOCALAPPDATA", ".")).joinpath(
                        "sharp3d", "compile_error.log").write_text(
                        traceback.format_exc(), encoding="utf-8")
                except Exception:
                    pass
                return self.predictor(img_resized, disparity_factor)

    def __call__(self, img_resized: torch.Tensor, disparity_factor: torch.Tensor):
        """Allow using SharpPredictor directly as predict_fn."""
        return self.predict(img_resized, disparity_factor)
