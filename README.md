# sharp3d

**平面照片 / 视频 → 立体 3D 桌面转换工具**

基于 Apple [SHARP](https://github.com/apple/ml-sharp) 单图 3D 高斯泼溅模型，将任意 2D 图片或视频转换为立体 3D 输出。支持 SBS 左右并排、2.5D 视差动画、批量处理，提供 PySide6 图形界面与命令行两种使用方式。

---

## 工作原理

```
输入 (图片/视频)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  SHARP 模型推理                                          │
│  DINOv2 ViT-L/16 双编码器 + SPN 深度网络                  │
│  → 118万个 3D 高斯 (NDC 空间)                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  后处理管线                                              │
│  时域稳定 → 边缘柔化 → 反投影 → 收敛卡尔曼 → 渲染        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  gsplat 高斯泼溅渲染                                     │
│  左眼 + 右眼批量光栅化 → linearRGB → sRGB                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
输出 (SBS 图片/视频/深度图/PLY)
```

---

## 功能特性

### SBS 立体转换

| 功能 | 说明 |
|------|------|
| 立体格式 | Full SBS、Half SBS、Full TB、Half TB、Cross Eyed、Anaglyph 红青 |
| 瞳距 (IPD) | 50–80mm 可调，默认 63mm |
| 收敛平面 | 0–100% 分位数控制（0=自动50%），卡尔曼滤波逐帧平滑 |
| 立体强度 | 0.2–3.0× 倍率 |
| 输出分辨率 | 1.0× / 0.75× / 0.5× / 0.25× 或自定义宽度 |
| 视频编码 | H.264 / H.265 / AV1（NVENC GPU 优先，自动回退 CPU） |
| 音频 | 自动保留原始音轨 |
| HDR | 自动检测 HDR 输入 → 色调映射；可输出 HDR10 (10-bit PQ BT.2020) |
| 批量处理 | 拖入文件夹，逐文件转换，支持断点续传 |

### 时域稳定（防闪烁）

| 模式 | 原理 | 额外耗时 |
|------|------|----------|
| 关闭 | 逐帧独立，无帧间约束 | 0 |
| 全局对齐 | 最小二乘 scale-shift 对齐 + EMA 混合 | ~1ms/帧 |
| 自适应 | 逐像素置信度加权 EMA，保护运动物体 | ~2ms/帧 |
| 光流 | RAFT 双向光流 + 遮挡感知混合 | ~50ms/帧 |

附加处理（独立于上述模式）：

- **收敛卡尔曼平滑**（始终启用）：恒速模型卡尔曼滤波消除收敛平面逐帧跳动
- **属性 EMA**：opacity / scale 帧间指数平滑（FP16 存储，节省 19MB VRAM）
- **边缘柔化**（可选）：Sobel 梯度检测 + sigmoid 权重 + 5×5 高斯模糊，仅作用于深度不连续处，减轻遮挡拉伸伪影（~3ms/帧）
- **场景切换检测**：残差突变自动重置时域状态

### 2.5D 视差动画

| 轨迹 | 效果 |
|------|------|
| 横扫 (swipe) | 水平平移视差 |
| 摇晃 (shake) | 左右往复摆动 |
| 环绕 (rotate) | 绕场景中心旋转 |
| 前推 (rotate_forward) | 推进 + 旋转 |

参数：视差幅度 0.02–0.25、缩放 0–0.4、帧数 10–240、循环 1–5 次、播放帧率 24/30/60fps。

### 高斯查看器

独立窗口交互式 3D 查看器：鼠标拖拽旋转、滚轮缩放、拖入 PLY 文件加载、~30fps 实时渲染。

### 导出选项

| 选项 | 说明 |
|------|------|
| 深度图 | 伪彩色可视化（近=暖色，远=冷色）；视频输出为独立 H.264 文件 |
| PLY 高斯 | 逐帧导出编号序列 `{stem}_{00000}.ply`，兼容内置查看器 |

---

## 性能

测试环境：i9-13900HX + RTX 5070 Ti Laptop 12GB，4K 输入 → 7680×2160 SBS 输出

| 指标 | 数值 |
|------|------|
| 端到端单帧 | ~0.7s（~1.4 fps，含解码/编码） |
| SHARP 推理 (TRT FP16) | ~512ms |
| SHARP 推理 (TRT INT8) | ~340ms（speed 模式） |
| 300 帧 4K 视频 | ~3.5 分钟 |
| GPU 计算利用率 | ~83% |
| VRAM 占用 | ~4.3GB |

### 优化技术栈

| 优化 | 加速比 | 说明 |
|------|--------|------|
| ORT TensorRT FP16 | 1.74× | DINOv2 双编码器 (418ms → 240ms) |
| torch.compile max-autotune | 1.16× | 持久缓存，热启动 10.8s |
| GPU 四元数 (Shepperd) | 884× | 替代 scipy CPU 实现 |
| 解析法特征分解 | 3× | 替代 GPU SVD (省 0.4s/帧) |
| 3 级流水线 | — | 解码预取 / GPU 计算 / 编码重叠 |
| 异步 H2D 传输 | — | 侧 CUDA 流，与编码重叠 |
| NVENC 硬件编码 | — | 独立编码引擎，不占 CUDA 核心 |
| IO Binding 零拷贝 | — | GPU→ORT→GPU 无 CPU 往返 |

---

## 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 |
| GPU | NVIDIA RTX 30/40/50 系列（SM 8.0+），≥8GB VRAM |
| Python | 3.13 |
| CUDA | 13.0 |
| 编译器 | Visual Studio 2022（torch.compile 需要 cl.exe） |
| 磁盘 | ~15GB（模型权重 + 编译缓存） |

---

## 安装

```bash
# 1. 创建虚拟环境
python -m venv sharp3d-env
sharp3d-env\Scripts\activate

# 2. 安装 PyTorch (CUDA 13.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 3. 安装 gsplat（从 GitHub Releases 下载预编译 whl）
#    支持 SM 8.0 / 8.9 / 12.0 三种架构
pip install gsplat-*.whl
# 或从源码编译：
pip install git+https://github.com/nerfstudio-project/gsplat.git

# 4. 安装 SHARP 模型（editable）
git clone https://github.com/apple/ml-sharp
pip install -e ml-sharp

# 5. 安装其余依赖
pip install PySide6 onnxruntime-gpu pynvml imageio imageio-ffmpeg \
    pillow numpy plyfile triton-windows tensorrt
```

完整依赖清单见 [requirements.txt](requirements.txt)。

---

## 使用

### 图形界面

需先配置 MSVC 环境（torch.compile 需要 cl.exe）：

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set PYTHONPATH=<path-to>\sharp3d\src
python -m sharp3d.gui
```

首次启动约 90 秒（模型下载 + TensorRT 引擎构建 + torch.compile），后续启动约 15 秒（持久缓存）。

### 命令行

```bash
# 图片 → SBS
python -m sharp3d.cli input.png -o output.png

# 视频 → SBS（H.265 编码）
python -m sharp3d.cli input.mp4 -o output.mp4 --codec h265 --crf 18

# 强制 HDR10 输出
python -m sharp3d.cli input.mp4 --hdr

# 指定立体格式
python -m sharp3d.cli input.png --format anaglyph

# 使用 SVD 分解（参考精度）
python -m sharp3d.cli input.png --decompose svd
```

### INT8 校准（可选，speed 模式）

```bash
python -m sharp3d.calibrate_int8 --video representative.mp4 --frames 50
```

---

## 项目结构

```
src/sharp3d/
├── options.py         转换参数数据类 (ConvertOptions)
├── predict.py         模型加载 + FP16 + compile + 预热
├── ort_engine.py      ONNX Runtime TensorRT 加速引擎
├── pipeline.py        端到端管线（图片模式）
├── conversion.py      视频转换引擎（时域稳定 + 边缘柔化 + 卡尔曼收敛）
├── temporal.py        时域稳定器（全局/自适应/光流 + 属性 EMA）
├── render.py          gsplat 批量 SBS 渲染 + 单视角 + 深度图
├── unproject.py       NDC → 世界空间高斯反投影
├── quaternion.py      GPU 四元数转换（Shepperd 法）
├── eigendecompose.py  3×3 对称矩阵特征分解（解析法 + SVD）
├── formats.py         6 种立体格式打包
├── video.py           视频读写 + 编码器探测 + 音频混流
├── hdr.py             HDR 检测 + 色调映射 + HDR10 编码
├── calibrate_int8.py  TensorRT INT8 校准表生成
├── cli.py             命令行入口
└── gui/
    ├── main_window.py 主窗口（多进程架构，GPU 崩溃隔离）
    ├── sbs_tab.py     SBS 立体转换标签页
    ├── anim_tab.py    2.5D 视差动画标签页
    ├── gaussian_tab.py 高斯查看器（独立窗口）
    ├── worker.py      GPU 工作进程（3 级流水线）
    ├── widgets.py     自定义控件（进度条/GPU 监控/预览等）
    └── theme.py       系统主题检测 + 红青配色
```

---

## 已知限制

- **HDR 为格式级支持**：SHARP 是 SDR 模型（sRGB 输入），HDR 输入先色调映射为 SDR 再处理。输出的 HDR10 文件能在 HDR 设备正确显示（10-bit PQ BT.2020），但无法还原原片高光与广色域。
- **内部分辨率固定 1536×1536**：SHARP 架构约束（SPN 三级金字塔），与输入分辨率无关。
- **高斯数量固定 ~118 万**：模型固定输出，不可调节。
- **首次启动较慢**：TensorRT 引擎构建 + torch.compile 约 90 秒，后续有持久缓存（~15 秒）。

---

## 许可

本项目代码供个人学习研究使用。SHARP 模型权重及其源码遵循 [Apple 原始许可协议](https://github.com/apple/ml-sharp/blob/main/LICENSE)。
