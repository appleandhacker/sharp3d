"""批次 4（资源与维护性）验证脚本。

运行: sharp3d-env/Scripts/python.exe tools/check_batch4.py
覆盖:
  01 video import 副作用延后（IMAGEIO_FFMPEG_EXE 锁存）   ← 必须最先跑
  02 profiling 运行时读取 + 空帧 flush 清 ORT 累计
  03 _ENCODERS 探测失败不缓存 + 恢复 + 单飞锁
  04 crf clamp（encoder_output_params + Hdr10Writer 命令捕获）
  05 CLI 三项校验（crf/ipd/keyframe-interval → exit 2）
  06 VideoWriter 真实编码 + abort() 清理
  07 VideoReader 元数据 / 缺失文件 / 坏文件 / 幂等 close
  08 render_vr 相机缓存量化 + FIFO 上限
  09 formats._squeeze 新旧实现字节级一致
  10 projection 面选择新旧映射全网格一致 + plan 覆盖完整
  11 conversion._get_unprojection 按值缓存 + reset 清空
  12 ViTCapture.__getattr__ 抛 AttributeError 而非 KeyError
  13 i18n 模板零缺失（扫描器）+ EN 模式返回英文
  14 _ascii_safe_trt_dir 行为 + ort_engine/predict 关键代码形态
  15 pipeline 失败路径清理代码形态（ok 标志 / 排空 / abort / join）
"""

import itertools
import os
import subprocess
import sys
import tempfile
import threading
import traceback
import types
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
# 01  video import 副作用延后 — 必须是第一个 import sharp3d.* 的检查
# ────────────────────────────────────────────────────────────────────
@check
def check01_video_import_side_effect():
    os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
    import sharp3d.video as V

    assert V._IMAGEIO_FFMPEG_PINNED is False, "import 期不应已锁存 pin"
    assert "IMAGEIO_FFMPEG_EXE" not in os.environ, \
        "import 期不应写 IMAGEIO_FFMPEG_EXE"
    V._pin_imageio_ffmpeg()
    assert V._IMAGEIO_FFMPEG_PINNED is True, "首次调用应锁存"
    assert os.environ.get("IMAGEIO_FFMPEG_EXE") == str(V.FFMPEG), \
        "锁存后环境变量应指向解析出的 FFMPEG"
    V._pin_imageio_ffmpeg()  # 幂等
    assert os.environ.get("IMAGEIO_FFMPEG_EXE") == str(V.FFMPEG)


# ────────────────────────────────────────────────────────────────────
# 02  profiling 运行时读取 + 空帧 flush 清 ORT 累计
# ────────────────────────────────────────────────────────────────────
@check
def check02_profiling_dynamic():
    import sharp3d.profiling as P

    saved = os.environ.get("SHARP3D_PROFILE")
    try:
        os.environ["SHARP3D_PROFILE"] = "0"
        assert P.ENABLED is False and P.enabled() is False
        os.environ["SHARP3D_PROFILE"] = "1"
        assert P.ENABLED is True and P.cpu_trace() is False
        os.environ["SHARP3D_PROFILE"] = "3"
        assert P.ENABLED is True and P.CPU_TRACE is True and P.cpu_trace() is True
        del os.environ["SHARP3D_PROFILE"]
        assert P.ENABLED is False
        try:
            P.NO_SUCH_ATTR
            raise AssertionError("__getattr__ 应对未知属性抛 AttributeError")
        except AttributeError:
            pass

        # 空帧窗口: flush 不打印、不同步 CUDA，但仍要清 ORT 累计
        P.add_ort("patch_encoder", 7.5)
        t = P.get_timer()
        t._frames.clear()
        t._wall.clear()
        t.flush()  # 空帧分支
        assert P.take_ort() == 0.0, "空帧 flush 必须清掉 ORT 累计"
    finally:
        if saved is None:
            os.environ.pop("SHARP3D_PROFILE", None)
        else:
            os.environ["SHARP3D_PROFILE"] = saved


# ────────────────────────────────────────────────────────────────────
# 03  _ENCODERS 探测失败不缓存
# ────────────────────────────────────────────────────────────────────
@check
def check03_encoders_probe_fail_not_cached():
    import sharp3d.hdr as H

    assert isinstance(H._ENCODERS_LOCK, type(threading.Lock())), "单飞锁缺失"
    saved = os.environ.get("IMAGEIO_FFMPEG_EXE")
    try:
        H._ENCODERS = None
        os.environ["IMAGEIO_FFMPEG_EXE"] = "Z:/definitely-not-ffmpeg.exe"
        enc = H.available_encoders()
        assert enc == set(), "探测失败应返回空集"
        assert H._ENCODERS is None, "探测失败不得写入缓存（旧代码永久退 CPU）"

        # 恢复后下一次调用重试并成功缓存
        if saved is None:
            os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
        else:
            os.environ["IMAGEIO_FFMPEG_EXE"] = saved
        enc2 = H.available_encoders()
        assert isinstance(enc2, set) and len(enc2) > 0, "真实 ffmpeg 应探测到编码器"
        assert H._ENCODERS is not None and len(H._ENCODERS) > 0, "成功探测应缓存"
        assert "libx264" in H._ENCODERS or "h264_nvenc" in H._ENCODERS
    finally:
        if saved is None:
            os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
        else:
            os.environ["IMAGEIO_FFMPEG_EXE"] = saved


def _quality_from_list(flat):
    """从扁平参数列表提取 (-qp / -crf / crf=) 的质量值。"""
    for i, tok in enumerate(flat):
        if tok in ("-qp", "-crf") and i + 1 < len(flat):
            return int(flat[i + 1])
        if isinstance(tok, str) and tok.startswith("crf="):
            return int(tok.split("=", 1)[1].split(":")[0])
    return None


# ────────────────────────────────────────────────────────────────────
# 04  crf clamp
# ────────────────────────────────────────────────────────────────────
@check
def check04_crf_clamp():
    import sharp3d.hdr as H
    from sharp3d.video import encoder_output_params

    # CPU 链与 NVENC 链的输出参数都要夹紧（av1_nvenc 的 qp 走 0-255 AV1
    # 刻度，crf×5 映射；其余编码器 0-51 直传）
    for codec_lib, scale in (("libx264", 1), ("h264_nvenc", 1),
                             ("hevc_nvenc", 1), ("libx265", 1),
                             ("av1_nvenc", 5)):
        for crf_in, want in ((99, 51), (-5, 0), (18, 18), (40, 40)):
            params = encoder_output_params(codec_lib, crf_in, "medium")
            q = _quality_from_list(params)
            assert q is not None, f"{codec_lib}: 未找到质量参数 in {params}"
            assert q == min(want, 51) * scale, \
                f"{codec_lib} crf={crf_in} → {q} (期望 {min(want, 51) * scale})"

    # Hdr10Writer: 假 Popen 捕获命令行，验证 99 → 51、-5 → 0
    real_popen = H.subprocess.Popen
    captured = {}

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        raise RuntimeError("sentinel-stop-before-spawn")

    tmp = Path(tempfile.mkdtemp(prefix="sharp3d_chk04_"))
    for crf_in, want in ((99, 51), (-5, 0)):
        captured.clear()
        H.subprocess.Popen = fake_popen
        try:
            try:
                H.Hdr10Writer(str(tmp / f"h{crf_in}.mp4"), 64, 64, 24.0,
                              codec="h265", crf=crf_in)
                raise AssertionError("假 Popen 应触发 sentinel 异常")
            except RuntimeError as e:
                assert "sentinel" in str(e)
        finally:
            H.subprocess.Popen = real_popen
        q = _quality_from_list(captured["cmd"])
        assert q is not None, f"Hdr10Writer cmd 缺质量参数: {captured['cmd']}"
        assert q == want, f"Hdr10Writer crf={crf_in} → {q} (期望 {want})"


# ────────────────────────────────────────────────────────────────────
# 05  CLI 校验
# ────────────────────────────────────────────────────────────────────
@check
def check05_cli_validation():
    from sharp3d import cli

    def run(argv):
        saved = sys.argv
        sys.argv = ["sharp3d"] + argv
        try:
            cli.main()
        except SystemExit as e:
            return e.code
        finally:
            sys.argv = saved
        return None  # 正常返回（不应发生）

    bad_cases = [
        (["in.mp4", "--crf", "99"], 2),
        (["in.mp4", "--crf", "-1"], 2),
        (["in.mp4", "--ipd", "0"], 2),
        (["in.mp4", "--ipd", "-0.5"], 2),
        (["in.mp4", "--keyframe-interval", "0"], 2),
        (["in.mp4", "--keyframe-interval", "-3"], 2),
    ]
    for argv, want in bad_cases:
        got = run(argv)
        assert got == want, f"{argv} → exit {got} (期望 {want})"
    # 合法默认值应通过校验、走到输入存在性检查（缺文件 → exit 1）
    got = run(["definitely_missing_input.mp4"])
    assert got == 1, f"合法参数应通过校验并因缺输入 exit 1（实际 {got}）"


# ────────────────────────────────────────────────────────────────────
# 06  VideoWriter 真实编码 + abort 清理
# ────────────────────────────────────────────────────────────────────
@check
def check06_video_writer_roundtrip_and_abort():
    import sharp3d.video as V

    tmp = Path(tempfile.mkdtemp(prefix="sharp3d_chk06_"))
    out = tmp / "roundtrip.mp4"

    w = V.VideoWriter(str(out), fps=24.0, width=320, height=240,
                      codec="h264", crf=99)  # 走 clamp
    rng = np.random.default_rng(0)
    for _ in range(5):
        w.append_frame(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    w.close()
    assert out.exists() and out.stat().st_size > 1000, "close 后应有成片"
    assert not w.tmp_path.exists(), "close 后 tmp 应已改名"

    out2 = tmp / "aborted.mp4"
    w2 = V.VideoWriter(str(out2), fps=24.0, width=320, height=240,
                       codec="h264", crf=18)
    w2.append_frame(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    w2.abort()  # 不得抛异常
    assert not w2.tmp_path.exists(), "abort 后 tmp 应删除"
    assert not out2.exists(), "abort 不应产出成片"

    # 存成全局供 check07 复用
    global _CHK_VIDEO
    _CHK_VIDEO = out


_CHK_VIDEO = None


# ────────────────────────────────────────────────────────────────────
# 07  VideoReader
# ────────────────────────────────────────────────────────────────────
@check
def check07_video_reader():
    import sharp3d.video as V

    tmp = Path(tempfile.mkdtemp(prefix="sharp3d_chk07_"))

    # 缺失文件 → 异常（不得挂起、不得静默返回 0 尺寸对象）
    raised = False
    try:
        V.VideoReader(str(tmp / "no_such.mp4"))
    except Exception:
        raised = True
    assert raised, "缺失文件应抛异常"

    # 坏数据 → 异常（元数据读取路径的 except 分支 → close 后 re-raise）
    bad = tmp / "garbage.mp4"
    bad.write_bytes(b"\x00\x01not-a-video" * 512)
    raised = False
    try:
        V.VideoReader(str(bad))
    except Exception:
        raised = True
    assert raised, "坏文件应抛异常"

    # 真实文件（check06 产物）
    assert _CHK_VIDEO is not None and _CHK_VIDEO.exists()
    r = V.VideoReader(str(_CHK_VIDEO))
    assert (r.width, r.height) == (320, 240), f"尺寸 {(r.width, r.height)}"
    assert abs(r.fps - 24.0) < 0.5, f"fps {r.fps}"
    frm = r.get_frame(0)
    assert frm.shape == (240, 320, 3) and frm.dtype == np.uint8
    r.close()
    r.close()  # 幂等
    print(f"    [info] reader fps={r.fps} size={r.width}x{r.height}")


# ────────────────────────────────────────────────────────────────────
# 08  render_vr 相机缓存
# ────────────────────────────────────────────────────────────────────
@check
def check08_render_vr_cam_cache():
    import torch
    import sharp3d.render_vr as RV

    RV._CAM_CACHE.clear()
    calls = []

    def fake_get_cams(face_size, device, offset_vec):
        calls.append((face_size, offset_vec.clone()))
        return ("fake-cams", face_size, float(offset_vec[0]))

    orig = RV.get_cubemap_cameras
    RV.get_cubemap_cameras = fake_get_cams
    try:
        dev = torch.device("cpu")
        eo = torch.tensor([0.06315, 0.0, 0.0])

        c1 = RV._cameras_cached(64, dev, eo, 0.06312)  # round → 0.063
        c2 = RV._cameras_cached(64, dev, eo, 0.06309)  # 同桶 → 命中
        assert c1 is c2, "量化后同桶应命中缓存"
        assert len(calls) == 1, "命中不应重建相机"
        assert float(calls[0][1][1]) == 0.0 and float(calls[0][1][2]) == 0.0, \
            "y/z 应保持原样（不能 full_like 全填）"
        # float32 张量: 0.063 存为 0.0630000014…，用容差比较
        assert abs(float(calls[0][1][0]) - 0.063) < 1e-6, \
            f"重建的 x 应为量化值 0.063（fp32）, 实际 {float(calls[0][1][0])!r}"

        RV._cameras_cached(96, dev, eo, 0.06312)  # 不同 face_size → 新条目
        assert len(calls) == 2

        # 填满并越过上限 → 有界 + FIFO 逐出
        keys_before = list(RV._CAM_CACHE)
        for i in range(10):
            RV._cameras_cached(200 + i, dev, eo, 0.06312)
        assert len(RV._CAM_CACHE) <= RV._CAM_CACHE_MAX, \
            f"缓存无上限增长: {len(RV._CAM_CACHE)}"
        assert keys_before[0] not in RV._CAM_CACHE, "最老条目应被 FIFO 逐出"
    finally:
        RV.get_cubemap_cameras = orig
        RV._CAM_CACHE.clear()


# ────────────────────────────────────────────────────────────────────
# 09  formats._squeeze 新旧字节级一致
# ────────────────────────────────────────────────────────────────────
@check
def check09_formats_squeeze_equiv():
    import torch
    import torch.nn.functional as F
    from sharp3d.formats import _squeeze

    def old_squeeze(img, size):
        """HEAD 版本（git show HEAD:src/sharp3d/formats.py 原文）。"""
        t = img.permute(2, 0, 1)[None].float()
        out = F.interpolate(t, size=size, mode="bilinear", align_corners=True)
        return out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)

    g = torch.Generator().manual_seed(7)
    cases = [((37, 53), (23, 41)), ((64, 64), (128, 128)),
             ((240, 320), (120, 160)), ((9, 7), (5, 3)),
             ((1536, 1536), (768, 768)), ((1536, 1536), (1536, 768))]
    for (h, w), (th, tw) in cases:
        img = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8,
                            generator=g)
        a = old_squeeze(img, (th, tw))
        b = _squeeze(img, (th, tw))
        assert a.dtype == b.dtype == torch.uint8
        assert a.shape == b.shape
        assert torch.equal(a, b), f"{(h, w)}→{(th, tw)} 字节不一致 " \
                                  f"(diff {(a != b).sum().item()})"
    # 通道数泛化（1 / 4 通道同样逐通道独立）
    for c in (1, 4):
        img = torch.randint(0, 256, (20, 31, c), dtype=torch.uint8,
                            generator=g)
        assert torch.equal(old_squeeze(img, (11, 17)),
                           _squeeze(img, (11, 17))), f"{c} 通道不一致"
    print("    [info] 6 组尺寸 + 1/4 通道全部字节一致")


# ────────────────────────────────────────────────────────────────────
# 10  projection 面选择：新旧映射一致 + plan 覆盖完整
# ────────────────────────────────────────────────────────────────────
@check
def check10_projection_face_equiv():
    import torch
    import sharp3d.projection as PJ

    def face_map_new(x, y, z):
        """当前源码（projection.py L607-614）逐字。"""
        abs_x, abs_y, abs_z = x.abs(), y.abs(), z.abs()
        face_x = (abs_x >= abs_y) & (abs_x >= abs_z)
        face_y = ~face_x & (abs_y >= abs_z)
        return torch.where(
            face_x,
            torch.where(x >= 0, 0, 1),
            torch.where(face_y, torch.where(y >= 0, 2, 3),
                        torch.where(z >= 0, 4, 5)),
        )

    def face_map_old(x, y, z):
        """HEAD 版本（git show HEAD:src/sharp3d/projection.py L602-608）逐字。"""
        abs_x, abs_y, abs_z = x.abs(), y.abs(), z.abs()
        out = torch.zeros(x.shape[0], dtype=torch.long)
        out[(x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)] = 0
        out[(x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)] = 1
        out[(y > 0) & (abs_y > abs_x) & (abs_y >= abs_z)] = 2
        out[(y < 0) & (abs_y > abs_x) & (abs_y >= abs_z)] = 3
        out[(z > 0) & (abs_z > abs_x) & (abs_z > abs_y)] = 4
        out[(z < 0) & (abs_z > abs_x) & (abs_z > abs_y)] = 5
        return out

    # 随机方向 + 大量并列（tie）方向 + 退化方向
    g = torch.Generator().manual_seed(42)
    rnd = torch.randn(4096, 3, generator=g)
    rnd[torch.norm(rnd, dim=1) == 0] = torch.tensor([1.0, 0.0, 0.0])
    ties = []
    for a in (0.3, 1.0, 2.0):
        for b in (0.5, 1.0):
            for perm in itertools.permutations([a, a, b]):
                for signs in itertools.product((1, -1), repeat=3):
                    ties.append([perm[i] * signs[i] for i in range(3)])
    ties += [[0, 0, 0], [1, 1, 1], [1, 1, -1], [0, 1, 1], [1, 0, 1],
             [1, -0.0, 0], [-0.0, -0.0, -0.0]]
    dirs = torch.cat([rnd, torch.tensor(ties, dtype=torch.float32)])
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    old_map, new_map = face_map_old(x, y, z), face_map_new(x, y, z)
    n_diff = (old_map != new_map).sum().item()
    assert n_diff == 0, f"新旧面映射在 {n_diff} 个方向上不一致"

    # 模块集成: _build_equirect_plan 的 (face, u, v) 自洽 + 全覆盖
    W, H = 128, 64
    plan = PJ._build_equirect_plan(W, H, False, torch.device("cpu"))
    n_pix = W * H
    seen = torch.zeros(n_pix, dtype=torch.bool)
    ones = None
    for fi, idx32, grid in plan:
        idx = idx32.to(torch.long)
        uv = grid[0, 0]
        u, v = uv[:, 0], uv[:, 1]
        ones = torch.ones_like(u)
        if fi == 0:
            dd = torch.stack([ones, v, -u], dim=1)
        elif fi == 1:
            dd = torch.stack([-ones, v, u], dim=1)
        elif fi == 2:
            dd = torch.stack([u, ones, -v], dim=1)
        elif fi == 3:
            dd = torch.stack([u, -ones, v], dim=1)
        elif fi == 4:
            dd = torch.stack([u, v, ones], dim=1)
        else:
            dd = torch.stack([-u, v, -ones], dim=1)
        pred = face_map_new(dd[:, 0], dd[:, 1], dd[:, 2])
        bad = (pred != fi).sum().item()
        assert bad == 0, f"face {fi}: {bad} 个像素的 (u,v) 与面选择不自洽"
        seen[idx] = True
    assert bool(seen.all()), "plan 未覆盖全部像素"
    print(f"    [info] {len(dirs)} 方向新旧一致; plan {W}x{H} 全覆盖且自洽")


# ────────────────────────────────────────────────────────────────────
# 11  conversion._get_unprojection 按值缓存
# ────────────────────────────────────────────────────────────────────
@check
def check11_unprojection_value_cache():
    import numpy as np
    import torch
    from sharp3d.conversion import VideoConversionEngine
    from sharp3d.unproject import prepare_input

    # 用真实生产路径构造 ir（(4,4) 内参，帧帧新对象）
    _, _, ir1, _ = prepare_input(
        np.zeros((240, 320, 3), dtype=np.uint8), 500.0,
        torch.device("cpu"))
    assert ir1.shape == (4, 4), f"prepare_input ir 形状 {tuple(ir1.shape)}"

    eng = object.__new__(VideoConversionEngine)  # 绕过重型 __init__
    eng._eye4 = torch.eye(4)
    eng._unproj = None
    eng._unproj_ir = None

    m1 = eng._get_unprojection(ir1)
    assert eng._unproj_ir is not None and eng._unproj_ir is not ir1, \
        "应按值克隆缓存 ir"
    m2 = eng._get_unprojection(ir1.clone())  # 新对象、同值
    assert m2 is m1, "同值不同对象应命中缓存（旧 identity 判定会 miss）"
    ir2 = ir1.clone()
    ir2[0, 0] = float(ir2[0, 0]) * 0.9
    m3 = eng._get_unprojection(ir2)
    assert m3 is not m1, "不同值应重建"
    assert torch.equal(eng._unproj_ir, ir2), "缓存 ir 应更新为新值"

    # reset() 清空（stub 掉 reset() 触及的时域稳定器）
    eng._stab = types.SimpleNamespace(reset=lambda: None)
    eng._conv_kf = types.SimpleNamespace(reset=lambda: None)
    eng.reset()
    assert eng._unproj is None and eng._unproj_ir is None, \
        "reset() 必须清掉 unprojection 缓存（批处理换视频错位修复）"


# ────────────────────────────────────────────────────────────────────
# 12  ViTCapture.__getattr__
# ────────────────────────────────────────────────────────────────────
@check
def check12_vitcapture_attrerror():
    import torch.nn as nn
    from sharp3d.spn_tail import ViTCapture

    v = ViTCapture(nn.Identity())
    out = v.forward(__import__("torch").zeros(2))
    assert out is not None and v.last is not None, "正常委托应工作"

    # __init__ 未执行的半构造对象：必须抛带说明的 AttributeError，而非 KeyError
    bare = object.__new__(ViTCapture)
    try:
        bare.inner
        raise AssertionError("裸对象访问 .inner 应抛 AttributeError")
    except AttributeError as e:
        assert "inner" in str(e), f"异常信息应含属性名: {e}"
    try:
        bare.any_random_attr
        raise AssertionError("裸对象访问任意属性应抛 AttributeError")
    except AttributeError:
        pass


# ────────────────────────────────────────────────────────────────────
# 13  i18n 模板
# ────────────────────────────────────────────────────────────────────
@check
def check13_i18n_templates():
    # AST 扫描器：占位符模板零缺失
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "_scan_tr.py")],
        cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"扫描器退出码 {r.returncode}: {r.stderr[-500:]}"
    assert "MISSING: 0" in r.stdout, f"扫描报告:\n{r.stdout[-1500:]}"

    import sharp3d.gui.i18n as I

    old = I._LANG
    I._LANG = "en"
    try:
        assert I.tr('文件出错已跳过 · {}') == 'File failed, skipped · {}'
        assert I.tr('批量完成 · 共 {} 个文件 · 跳过 {} 个失败') == \
            'Batch finished · {} files total · {} skipped'
        assert I.tr('批量转换完成 · {} 个文件 · 跳过 {} 个失败') == \
            'Batch conversion finished · {} files · {} skipped'
        assert I.tr('批量转换完成 · {} 个文件 · 跳过 {} 个失败: {}') == \
            'Batch conversion finished · {} files · {} skipped: {}'
    finally:
        I._LANG = old
    assert I.tr('文件出错已跳过 · {}') == '文件出错已跳过 · {}', \
        "zh 模式应原样返回"


# ────────────────────────────────────────────────────────────────────
# 14  TRT 路径重定向 + 关键代码形态
# ────────────────────────────────────────────────────────────────────
@check
def check14_trt_dir_and_code_shape():
    from sharp3d.ort_engine import _ascii_safe_trt_dir

    ascii_base = Path(tempfile.gettempdir()) / "sharp3d_chk14_ascii" / "trt_v3"
    assert _ascii_safe_trt_dir(ascii_base) == ascii_base, \
        "ASCII 路径应原样返回（且不建目录）"

    non_ascii = Path("C:/临时_目录_测试_sharp3d") / "trt_v3"
    out = _ascii_safe_trt_dir(non_ascii)
    assert out != non_ascii and str(out).isascii(), "非 ASCII 应重定向"
    assert out == Path(os.environ.get("ProgramData", r"C:\ProgramData")) / \
        "sharp3d" / "trt_cache", f"重定向目标应为 ProgramData: {out}"

    predict_src = (REPO / "src" / "sharp3d" / "predict.py").read_text("utf-8")
    ort_src = (REPO / "src" / "sharp3d" / "ort_engine.py").read_text("utf-8")
    assert '_trt_dir = _ascii_safe_trt_dir(cache_dir / "trt_v3")' in predict_src
    assert "trt_cached = _trt_dir.exists() and any(_trt_dir.glob(\"*.engine\"))" \
        in predict_src, "缓存探测必须走重定向后的目录"
    assert "_trt_probe_dir" in predict_src, "启动日志引擎清单也应重定向"
    assert 'cache_base / f"ws{self._workspace_gb}"' in ort_src, \
        "ORTEncoder 缓存须按 workspace 分目录"
    assert 'f"{label}_ws{workspace_gb}"' in ort_src, \
        "ORTRestSession 缓存须按 workspace 分目录"
    assert "self._workspace_gb * 1024 * 1024 * 1024" in ort_src, \
        "workspace 大小须真实传入 trt_max_workspace_size"


# ────────────────────────────────────────────────────────────────────
# 15  pipeline 失败路径清理形态
# ────────────────────────────────────────────────────────────────────
@check
def check15_pipeline_cleanup_shape():
    src = (REPO / "src" / "sharp3d" / "pipeline.py").read_text("utf-8")
    for needle in (
        "ok = False",
        "frame_q.get_nowait()",
        "writer.abort()",
        "decoder.join(timeout=5)",
        "except queue.Empty:",
    ):
        assert needle in src, f"pipeline.py 缺少: {needle}"
    # finally 块顺序: 排空队列 → abort →（后续）join
    # （abort 的首次出现可能在 _AsyncFrameSink 的注释里，从 finally 起搜）
    i_finally = src.index("        finally:\n            # Unblock the decoder")
    i_abort = src.index("writer.abort()", i_finally)
    i_join = src.index("decoder.join(timeout=5)", i_abort)
    assert i_finally < i_abort < i_join, "finally(排空+abort) 应在 join 之前"


# ────────────────────────────────────────────────────────────────────
def main():
    print(f"check_batch4: {len(CHECKS)} 项检查  (repo={REPO})")
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
