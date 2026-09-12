"""Produce three same-source sample videos for human quality judgement:
    _cmp_default.mp4  — 默认路径（fp16 ViT TRT + torch.compile SPN）
    _cmp_trtfp16.mp4  — 三会话 TRT（尾部 fp16 引擎）
    _cmp_int8.mp4     — 三会话 TRT（INT8 QDQ 编码器）
Each run is a separate subprocess (env-isolated pipeline selection); wall
time per mode is recorded for the conversion-time comparison.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT.parent / "sharp3d-env" / "Scripts" / "python.exe")

runs = [
    ("default", {}, "0"),
    ("trtfp16", {"SHARP3D_FULL_TRT": "1", "SHARP3D_FULL_TRT_TAILFP16": "1"}, "0"),
    ("int8", {"SHARP3D_FULL_TRT": "1"}, "1"),
]

# INT8 需要 QDQ 模型（此前清理过）——缺失则重新量化
qdq = ROOT / ".cache" / "onnx" / "patch_encoder_int8.onnx"
if not qdq.exists():
    print("=== 重新生成 INT8 QDQ 模型（MinMax, 24 帧）… ===", flush=True)
    t0 = time.time()
    subprocess.run([PY, str(ROOT / "tests" / "quantize_vit_int8.py"),
                    "--frames", "24", "--method", "MinMax"],
                   cwd=str(ROOT), check=True)
    print(f"=== 量化完成 ({time.time()-t0:.0f}s) ===", flush=True)

times = {}
for mode, extra_env, int8 in runs:
    env = dict(os.environ)
    env.update(extra_env)
    print(f"\n=== 转换 {mode} (env: {extra_env or '默认'}) ===", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, str(ROOT / "tests" / "_convert_one.py"),
                        mode, int8],
                       cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
    total = time.time() - t0
    ok = r.returncode == 0
    # 提取 ###TIME### 行
    line = next((l for l in (r.stdout or "").splitlines()
                 if l.startswith("###TIME###")), None)
    print(line or f"(无计时输出, returncode={r.returncode})")
    if not ok:
        print(r.stdout[-1500:] if r.stdout else "")
        print(r.stderr[-1500:] if r.stderr else "")
    times[mode] = (total, ok, line)

print("\n=== 耗时汇总（含模型加载） ===", flush=True)
for mode, (total, ok, line) in times.items():
    print(f"  {mode:8s} 总耗时 {total:7.1f}s  {'OK' if ok else 'FAIL'}")
    if line:
        print(f"           {line}")
