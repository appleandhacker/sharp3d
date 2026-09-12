"""Sustained FP16 / INT8 / FP8 GEMM throughput with thermal-cooldown isolation.

Each precision gets:
  1. a 120s cooldown (so every test starts from the same thermal state),
  2. warmup,
  3. a burst measurement (50 iters),
  4. a sustained 20s measurement sampled in 4 windows — a laptop GPU under
     continuous load clocks down, and a single 20-iter burst can hide that.

    sharp3d-env\\Scripts\\python.exe tests\\bench_precision.py
"""
import time

import torch

dev = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"(sm{torch.cuda.get_device_capability(dev)[0]}"
      f"{torch.cuda.get_device_capability(dev)[1]})  torch {torch.__version__}")

SHAPES = [(35 * 577, 1024, 3072, "qkv"), (35 * 577, 1024, 4096, "mlp.fc1")]
ITERS_BURST = 50
SUSTAIN_SECONDS = 20.0
WINDOWS = 4
COOLDOWN_S = 120


def cooldown(seconds: float, label: str) -> None:
    print(f"\n--- {label}: 散热 {seconds:.0f}s ---", flush=True)
    end = time.time() + seconds
    while time.time() < end:
        remain = end - time.time()
        print(f"  cooldown {remain:5.0f}s remain", flush=True)
        time.sleep(min(30, remain))
    t = torch.cuda.get_device_properties(dev).temperature if hasattr(
        torch.cuda.get_device_properties(dev), "temperature") else None
    print("  cooldown done", flush=True)


def bench(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def make_ops(prec: str):
    """Return {tag: callable} for the requested precision."""
    ops = {}
    for M, K, N, tag in SHAPES:
        flops = 2 * M * K * N
        if prec == "FP16":
            x = torch.randn(M, K, device=dev, dtype=torch.float16)
            w = torch.randn(N, K, device=dev, dtype=torch.float16) / K
            ops[tag] = (lambda a=x, b=w: torch.mm(a, b.t()), flops)
        elif prec == "INT8":
            x = torch.randint(-32, 32, (M, K), device=dev, dtype=torch.int8)
            w = torch.randint(-32, 32, (N, K), device=dev, dtype=torch.int8)
            ops[tag] = (lambda a=x, b=w: torch._int_mm(a, b.t()), flops)
        elif prec == "FP8":
            x = torch.randn(M, K, device=dev, dtype=torch.float16)
            w = torch.randn(N, K, device=dev, dtype=torch.float16) / K
            x8 = x.to(torch.float8_e4m3fn)
            w8 = w.to(torch.float8_e4m3fn)
            sa = torch.ones((), device=dev)
            sb = torch.ones((), device=dev)
            ops[tag] = (lambda a=x8, b=w8, s1=sa, s2=sb: torch._scaled_mm(
                a, b.t(), scale_a=s1, scale_b=s2, out_dtype=torch.float16),
                flops)
    return ops


print(f"\n每档精度: 预热 → 突发 {ITERS_BURST} 次 → 持续 {SUSTAIN_SECONDS:.0f}s"
      f"（{WINDOWS} 个窗口观察降频）", flush=True)

results = {}
for prec in ("FP16", "INT8", "FP8"):
    cooldown(COOLDOWN_S, prec)
    ops = make_ops(prec)

    print(f"\n=== {prec} 突发（{ITERS_BURST} 次均值） ===", flush=True)
    for tag, (fn, flops) in ops.items():
        t = bench(fn, ITERS_BURST)
        results.setdefault(prec, {})[tag] = {"burst_tops": flops / t / 1e12}
        print(f"  {tag:8s} {t*1e3:7.2f}ms  {flops/t/1e12:7.1f} TFLOPS", flush=True)

    print(f"=== {prec} 持续 {SUSTAIN_SECONDS:.0f}s（{WINDOWS} 窗口） ===", flush=True)
    for tag, (fn, flops) in ops.items():
        win = SUSTAIN_SECONDS / WINDOWS
        counts = []
        t_end = time.time() + SUSTAIN_SECONDS
        t_win = time.time() + win
        n = 0
        while time.time() < t_end:
            fn()
            n += 1
            if time.time() >= t_win:
                counts.append(n)
                n = 0
                t_win += win
        tops = [c * flops / win / 1e12 for c in counts]
        results[prec][tag]["sustain_tops"] = [round(v, 1) for v in tops]
        decay = (tops[-1] / tops[0] * 100) if tops[0] else 0
        print(f"  {tag:8s} " + "  ".join(f"{v:6.1f}" for v in tops)
              + f"  TFLOPS/窗   首末比 {decay:.0f}%", flush=True)

print("\n=== 汇总 ===", flush=True)
for prec in ("FP16", "INT8", "FP8"):
    for tag, d in results[prec].items():
        print(f"{prec:5s} {tag:8s} burst={d['burst_tops']:6.1f}  "
              f"sustain={d['sustain_tops']}", flush=True)
