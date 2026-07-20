# sharp3d

把平面照片 / 视频转换成立体 3D（SBS 左右并排）的桌面工具，基于 Apple 的
[SHARP](https://github.com/apple/ml-sharp) 单图 3D 高斯泼溅模型。

输入一张 2D 图片或视频 → SHARP 预测 3D 高斯场景 → 从左右眼两个视角渲染 →
输出 SBS 立体图片 / 视频。另含 2.5D 视差动画（复用 SHARP 的四种相机轨迹）。

## 功能特性

- **SBS 立体转换**：双固定视角渲染，瞳距 IPD / 收敛深度 / 立体强度可调
- **2.5D 视差动画**：横扫 / 摇晃 / 环绕 / 前推四种轨迹，可循环播放与导出
- **视频处理**：逐帧处理，H.264 / H.265 / AV1 编码，自动保留原始音轨
- **可选导出**：深度图可视化、PLY 高斯文件
- **PySide6 图形界面**：红青立体主题，色调跟随系统亮 / 暗模式，GPU 实时监控
- **响应式调参**：拖动滑块时从缓存高斯快速重渲染预览（预测与渲染分离）
- **HDR 支持**：自动检测 HDR 输入并色调映射后处理，可输出 HDR10（10-bit PQ BT.2020）

## 性能（RTX 5070 Ti 12GB，4K 输入）

| 指标 | 数值 |
|------|------|
| 单帧耗时 | 0.93s（1.07 fps） |
| 300 帧 4K 视频 | 约 4.7 分钟 |
| VRAM 占用 | 约 4.3GB |

关键优化：`torch.compile` + FP16 推理、GPU 四元数（Shepperd 法，替代 scipy）、
解析法 3×3 对称特征分解（替代 GPU SVD，省 0.4s/帧）、gsplat 批量双眼渲染。

## 依赖

- Python 3.13
- PyTorch 2.8+（CUDA 12.8，Blackwell sm_120 需此版本）
- [gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3（Windows 需 patch MSVC 编译标志）
- [ml-sharp](https://github.com/apple/ml-sharp)（SHARP 模型，editable 安装）
- `triton-windows<3.4`（Windows 下启用 torch.compile）
- PySide6、pynvml、imageio、Pillow、numpy、plyfile
- ffmpeg（视频编码与音频混流，需在 PATH）
- Visual Studio 2022（torch.compile 需要 cl.exe）

## 安装

```bash
# 1. 创建虚拟环境并安装 PyTorch (cu128)
python -m venv sharp3d-env
sharp3d-env\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. 安装 gsplat（Windows 需先 patch _backend.py 的 MSVC 标志）
pip install gsplat==1.5.3

# 3. 安装 SHARP（editable）
git clone https://github.com/apple/ml-sharp
pip install -e ml-sharp

# 4. 安装其余依赖
pip install PySide6 pynvml imageio pillow numpy plyfile "triton-windows<3.4"
```

## 使用

### 图形界面

需先配置 MSVC 环境（torch.compile 需要 cl.exe）：

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set PYTHONPATH=<path-to>\sharp3d\src
python -m sharp3d.gui
```

### 命令行

```bash
python -m sharp3d.cli input.png -o output.png          # 图片 → SBS
python -m sharp3d.cli input.mp4 -o output.mp4          # 视频 → SBS
python -m sharp3d.cli input.mp4 --codec h265 --crf 18  # 指定编码
python -m sharp3d.cli input.mp4 --hdr                  # 强制 HDR10 输出（HDR 输入自动启用）
python -m sharp3d.cli input.png --decompose svd        # 用 SVD 分解（参考）
```

## 项目结构

```
src/sharp3d/
├── quaternion.py      GPU 四元数转换（Shepperd 法）
├── eigendecompose.py  3×3 对称矩阵特征分解（解析法 + SVD）
├── unproject.py       NDC → 世界空间反投影
├── render.py          批量 SBS 渲染 + 单视角渲染 + 深度图
├── predict.py         模型加载 + compile + FP16 推理
├── pipeline.py        端到端管线
├── video.py           视频读写 + 音频混流
├── hdr.py             HDR 检测 + 色调映射 + HDR10 编码
├── cli.py             命令行入口
└── gui/
    ├── theme.py       系统主题检测 + 红青配色 + QSS
    ├── widgets.py     自定义控件（动画进度条 / GPU 监控 / 预览面板等）
    ├── worker.py      GPU 工作线程（懒加载管线）
    ├── sbs_tab.py     SBS 转换标签页
    ├── anim_tab.py    2.5D 视差动画标签页
    └── main_window.py 主窗口 + 状态栏
```

## 已知限制

- **HDR 为格式级支持，非真 HDR 还原**：SHARP 是 SDR 模型（sRGB 输入、
  linearRGB 输出、值域 0–1），3D 重建全程在 SDR 空间进行。因此 HDR 输入会先
  色调映射成 SDR 再处理，输出的 HDR10 文件动态范围 / 色域仍是 SDR 级别——
  它能在 HDR 设备上正确显示（10-bit、PQ、BT.2020，不发灰、无色带），但无法
  还原原片的高光与广色域。这是模型本质限制。
- 内部分辨率固定 1536×1536（SHARP 架构约束，SPN 三级金字塔要求）。
- 高斯数量固定约 118 万（模型固定输出）。

## 许可

本项目代码供个人学习研究使用。SHARP 模型权重及其源码遵循 Apple 的原始许可协议。
