"""CLI 管线性能优化（perf-1）验证脚本。

覆盖:
  01 render 尾链新旧实现字节级一致（逐眼 in-place vs 整帧链）
  02 unproject ir/df 主机缓存：同参同对象、异参不同对象、FIFO 上限
  03 conversion._get_unprojection 恒等快路径（同对象免 torch.equal）+ 值缓存不变
  04 _AsyncFrameSink：顺序写入、错误传播、shutdown 幂等
  05 Hdr10Writer.write_frame memoryview 等价（假 Popen 捕获 stdin 字节）
  06 pipeline.py 形态检查（sink 接入 / VideoReader 死导入已清）
  07 _soften_depth_edges：重复 sigmoid 已删 + kernel 缓存（源码形态）

运行: sharp3d-env/Scripts/python.exe tools/check_perf_cli.py
"""

import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ────────────────────────────────────────────────────────────────────
@check
def check01_render_tail_byte_equal():
    import torch

    def linearRGB2sRGB(x):
        THRESHOLD = 0.0031308
        low = x * 12.92
        high = 1.055 * x.clamp(min=THRESHOLD).pow(1.0 / 2.4) - 0.055
        return torch.where(x <= THRESHOLD, low, high)

    def old_tail(rc, w):
        rc = linearRGB2sRGB(rc)
        left = (rc[0] * 255).clamp(0, 255).to(torch.uint8)
        right = (rc[1] * 255).clamp(0, 255).to(torch.uint8)
        return torch.cat([left, right], dim=1)

    def new_tail(rc, w):
        h = rc.shape[1]
        sbs = torch.empty((h, w * 2, 3), dtype=torch.uint8, device=rc.device)
        for eye in range(2):
            img = linearRGB2sRGB(rc[eye])
            img.mul_(255.0).clamp_(0.0, 255.0)
            sbs[:, eye * w:(eye + 1) * w] = img.to(torch.uint8)
        return sbs

    g = torch.Generator().manual_seed(11)
    for h, w in ((37, 53), (240, 320), (216, 384)):
        # 线性 RGB 值域含 0/阈值两侧/超 1（渲染可能轻微过曝）
        rc = torch.rand((2, h, w, 3), generator=g) * 1.3
        a = old_tail(rc, w)
        b = new_tail(rc, w)
        assert a.dtype == b.dtype == torch.uint8 and a.shape == b.shape
        assert torch.equal(a, b), f"{(h, w)} 尾链字节不一致 " \
                                  f"({(a != b).sum().item()} 像素)"
    print("    [info] 3 组尺寸新旧尾链字节一致（含 >1.0 过曝与阈值两侧）")


# ────────────────────────────────────────────────────────────────────
@check
def check02_unproject_ir_cache():
    import torch
    from sharp3d.unproject import prepare_input, _IR_CACHE

    _IR_CACHE.clear()
    kw = dict(f_px=500.0, device=torch.device("cpu"))
    _, df1, ir1, _ = prepare_input(np.zeros((240, 320, 3), np.uint8), **kw)
    _, df2, ir2, _ = prepare_input(np.zeros((240, 320, 3), np.uint8), **kw)
    assert ir1 is ir2, "同参数应返回同一 ir 对象（恒等快路径的前提）"
    assert df1 is df2, "同参数应返回同一 df 对象"
    _, _, ir3, _ = prepare_input(np.zeros((240, 320, 3), np.uint8),
                                 f_px=650.0, device=torch.device("cpu"))
    assert ir3 is not ir1 and not torch.equal(ir3, ir1), "异 f_px 应不同"
    assert not torch.equal(ir3, ir1)
    for i in range(12):  # 越过 FIFO 上限
        prepare_input(np.zeros((240, 320, 3), np.uint8),
                      f_px=700.0 + i, device=torch.device("cpu"))
    assert len(_IR_CACHE) <= 8, f"缓存无上限: {len(_IR_CACHE)}"
    _IR_CACHE.clear()


# ────────────────────────────────────────────────────────────────────
@check
def check03_unproj_identity_fast_path():
    import torch
    from sharp3d.conversion import VideoConversionEngine

    eng = object.__new__(VideoConversionEngine)
    eng._eye4 = torch.eye(4)
    eng._unproj = None
    eng._unproj_ir = None

    ir = torch.tensor([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1],
                       [0, 0, 0, 1][:4]]) if False else torch.tensor(
        [[700.0, 0, 320, 0], [0, 700.0, 240, 0], [0, 0, 1, 0],
         [0, 0, 0, 1]])
    m1 = eng._get_unprojection(ir)
    assert eng._get_unprojection(ir) is m1, "同对象应走恒等快路径"
    m2 = eng._get_unprojection(ir.clone())
    assert m2 is m1, "同值不同对象仍应命中（值比较兜底）"
    ir2 = ir.clone()
    ir2[0, 0] = 650.0
    assert eng._get_unprojection(ir2) is not m1, "异值应重建"


# ────────────────────────────────────────────────────────────────────
@check
def check04_async_frame_sink():
    import torch
    from sharp3d.pipeline import _AsyncFrameSink

    class FakeWriter:
        def __init__(self, fail_on=None):
            self.frames = []
            self.fail_on = fail_on
            self._n = 0

        def append_frame(self, frame):
            self._n += 1
            if self.fail_on is not None and self._n == self.fail_on:
                raise RuntimeError("fake writer failure")
            self.frames.append(frame.copy())

        def write_frame(self, frame):
            self.append_frame(frame)

    # 正常路径: 顺序 + 内容一致
    w = FakeWriter()
    sink = _AsyncFrameSink(w, use_hdr=False)
    srcs = []
    for i in range(7):
        t = torch.randint(0, 256, (8, 16, 3), dtype=torch.uint8)
        srcs.append(t.clone())
        sink.submit(t)
    sink.finish()
    sink.shutdown()  # 幂等
    assert len(w.frames) == 7
    for a, b in zip(w.frames, srcs):
        assert np.array_equal(a, b), "帧内容或顺序不一致"

    # 错误传播: 第 3 帧写失败 → submit/finish 抛错
    w2 = FakeWriter(fail_on=3)
    sink2 = _AsyncFrameSink(w2, use_hdr=False)
    raised = None
    try:
        for i in range(6):
            sink2.submit(torch.randint(0, 256, (8, 16, 3), dtype=torch.uint8))
        sink2.finish()
    except RuntimeError as e:
        raised = e
    assert raised is not None, "写帧失败应传播到主线程"
    assert "失败" in str(raised)
    sink2.shutdown()
    sink2.shutdown()  # 幂等，不得抛

    print(f"    [info] sink 正常 7 帧顺序一致；错误传播 + 幂等 shutdown 通过")


# ────────────────────────────────────────────────────────────────────
@check
def check05_hdr_write_frame_memoryview():
    import sharp3d.hdr as H

    real_popen = H.subprocess.Popen
    captured = {}

    class FakeStdin:
        def __init__(self):
            self.chunks = []

        def write(self, b):
            self.chunks.append(bytes(b))
            return len(b)

        def close(self):
            pass

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return FakeProc()

    H.subprocess.Popen = fake_popen
    try:
        w = H.Hdr10Writer(str(REPO / "_chk_hdr_tmp.mp4"), 8, 6, 24.0,
                          codec="h265", crf=18)
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        frame[..., 0] = np.arange(48).reshape(6, 8) % 256
        w.write_frame(frame)
    finally:
        H.subprocess.Popen = real_popen
    got = b"".join(captured["stdin_chunks"]) if "stdin_chunks" in captured \
        else b"".join(w._proc.stdin.chunks)
    assert got == frame.tobytes(), "memoryview 写入字节应与 tobytes 一致"
    w.abort()
    (REPO / "_chk_hdr_tmp.mp4").unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────────────
@check
def check06_pipeline_shape():
    src = (REPO / "src" / "sharp3d" / "pipeline.py").read_text("utf-8")
    assert "from .video import VideoWriter" in src
    assert "VideoReader" not in src, "VideoReader 死导入应已清除"
    assert "sink.submit(packed)" in src and "sink.finish()" in src
    # 阻塞 D2H 仅允许存在于单帧 process_image 冷路径（热循环必须走 sink）
    assert src.count("sbs_np = packed.cpu().numpy()") == 1, \
        "视频热循环的阻塞 D2H 应已被异步 sink 取代（仅剩 process_image 一处）"
    assert "class _AsyncFrameSink" in src and "pin_memory=True" in src


@check
def check07_soften_shape():
    src = (REPO / "src" / "sharp3d" / "conversion.py").read_text("utf-8")
    assert src.count("edge_weight = torch.sigmoid") == 1, \
        "重复 sigmoid 应只剩一处"
    assert "_soften_kernel_key" in src and "self._soften_kernel" in src
    assert "cached_ir is ir or" in src, "恒等快路径缺失"


# ────────────────────────────────────────────────────────────────────
def main():
    print(f"check_perf_cli: {len(CHECKS)} 项检查  (repo={REPO})")
    failed = []
    for fn in CHECKS:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("-" * 60)
    if failed:
        print(f"RESULT: {len(CHECKS) - len(failed)}/{len(CHECKS)} PASS, "
              f"FAILED: {', '.join(failed)}")
        return 1
    print(f"RESULT: {len(CHECKS)}/{len(CHECKS)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
