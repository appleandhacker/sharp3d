"""Benchmark all SHARP optimizations: IO Binding, image_encoder ORT, channels_last.

Tests each optimization individually and in combination to measure the
actual speedup on the real SHARP predict pipeline.

Usage:
    cd sharp3d
    ../sharp3d-env/Scripts/python.exe -m sharp3d.bench_optimizations
"""
import os
import sys
import time
import logging
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "ml-sharp" / "src"))

# cuDNN for ORT
torch_lib = Path(__import__("torch").__file__).parent / "lib"
os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

# Persistent compile cache
cache = PROJECT_ROOT / ".cache"
cache.mkdir(exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_model(device, use_channels_last=False, use_overlap=True):
    """Load SHARP model with optional optimizations."""
    import torch
    from sharp.models import PredictorParams, create_predictor

    state_dict = torch.hub.load_state_dict_from_url(
        "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
        progress=False, map_location="cpu",
    )
    params = PredictorParams()
    params.monodepth.use_patch_overlap = use_overlap
    predictor = create_predictor(params)
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)

    if use_channels_last:
        for mod in predictor.modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                mod.weight.data = mod.weight.data.to(
                    memory_format=torch.channels_last)

    return predictor


def setup_ort_encoders(predictor, device):
    """Replace patch_encoder and image_encoder with ORT TensorRT versions."""
    from sharp3d.ort_engine import (
        create_ort_patch_encoder, create_ort_image_encoder)
    spn = predictor.monodepth_model.monodepth_predictor.encoder

    n_replaced = 0
    ort_enc = create_ort_patch_encoder(predictor, device)
    if ort_enc is not None:
        spn.patch_encoder = ort_enc
        n_replaced += 1

    ort_img = create_ort_image_encoder(predictor, device)
    if ort_img is not None:
        spn.image_encoder = ort_img
        n_replaced += 1

    return n_replaced


def benchmark_predict(predictor, device, n_warmup=3, n_runs=10):
    """Benchmark predict() performance."""
    import torch
    from sharp3d.unproject import INTERNAL_SHAPE

    dummy_img = torch.randn(1, 3, *INTERNAL_SHAPE, device=device)
    dummy_df = torch.tensor([1.0], device=device, dtype=torch.float32)

    compiled = torch.compile(predictor, mode="max-autotune", dynamic=False)

    # Warmup (includes compilation)
    logger.info("  编译中 (首次约1分钟)…")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(n_warmup):
            _ = compiled(dummy_img, dummy_df)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = compiled(dummy_img, dummy_df)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return times


def print_stats(label, times):
    import numpy as np
    avg = np.mean(times)
    std = np.std(times)
    fps = 1.0 / avg
    logger.info(f"  {label}: {avg*1000:.0f}ms ± {std*1000:.0f}ms ({fps:.1f} FPS)")


def main():
    import torch
    device = torch.device("cuda")
    from sharp3d.unproject import INTERNAL_SHAPE

    logger.info("=" * 60)
    logger.info("SHARP 优化基准测试")
    logger.info("=" * 60)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"输入: {INTERNAL_SHAPE}")
    logger.info(f"torch: {torch.__version__}")
    try:
        import onnxruntime as ort
        logger.info(f"onnxruntime: {ort.__version__}")
    except ImportError:
        logger.info("onnxruntime: 未安装")
    logger.info("")

    # ── Test 1: Baseline (PyTorch only, no ORT) ──────────────────────
    logger.info("[1/4] 基线（纯PyTorch, 35 patches, 无channels_last）")
    pred = load_model(device, use_channels_last=False, use_overlap=True)
    times = benchmark_predict(pred, device, n_warmup=2, n_runs=8)
    print_stats("基线", times)
    del pred
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ── Test 2: ORT IO Binding + image_encoder ORT ───────────────────
    logger.info("[2/4] ORT IO Binding + image_encoder TensorRT (35 patches)")
    pred = load_model(device, use_channels_last=False, use_overlap=True)
    n = setup_ort_encoders(pred, device)
    logger.info(f"  ORT 替换了 {n} 个编码器")
    times = benchmark_predict(pred, device, n_warmup=2, n_runs=8)
    print_stats("IO Binding", times)
    del pred
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ── Test 3: ORT + channels_last ──────────────────────────────────
    logger.info("[3/4] ORT IO Binding + channels_last (35 patches)")
    pred = load_model(device, use_channels_last=True, use_overlap=True)
    n = setup_ort_encoders(pred, device)
    logger.info(f"  ORT 替换了 {n} 个编码器")
    times = benchmark_predict(pred, device, n_warmup=2, n_runs=8)
    print_stats("ORT + channels_last", times)
    del pred
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ── Test 4: Speed mode (21 patches + ORT + channels_last) ─────────
    logger.info("[4/4] 速度模式（21 patches + ORT IO Binding + channels_last）")
    pred = load_model(device, use_channels_last=True, use_overlap=False)
    n = setup_ort_encoders(pred, device)
    logger.info(f"  ORT 替换了 {n} 个编码器")
    times = benchmark_predict(pred, device, n_warmup=2, n_runs=8)
    print_stats("速度模式", times)
    del pred
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    logger.info("")
    logger.info("=" * 60)
    logger.info("完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
