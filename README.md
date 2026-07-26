# sharp3d

**平面照片/视频 → 立体3D · 全景/鱼眼 → VR 立体3D 转换工具**

基于 Apple [SHARP](https://github.com/apple/ml-sharp) 单图 3D 高斯泼溅模型，将任意 2D 图片或视频转换为立体 3D 输出。v2.0 新增全景 VR 管线：180°/360° 等距柱状投影或鱼眼输入 → 立体 3D VR 等距柱状投影输出，支持 VR 头显直接播放。

---

## 工作原理

### SBS 立体管线

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

### VR 全景管线 (v2.0)

```
输入 (180°/360° 等距柱状 或 鱼眼)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  面提取 (重叠 112° FOV)                                  │
│  360° → 6 面 cubemap · 180° → 4 轴最优半球               │
│  FOV_SCALE=1.5 确保相邻面重叠覆盖                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  逐面 SHARP 推理 (GPU-direct, 无 PCIe 往返)              │
│  每面 → 118万高斯 → 世界坐标变换 → 角度裁剪/权重衰减      │
│  合并为全局高斯场景                                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  VR 立体渲染                                             │
│  平行相机 ±IPD/2 · cubemap 6面×2眼 · HiGS/标准光栅化     │
│  180° 跳过 -Z 面 (5面×2眼) · 球面映射 → 等距柱状         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
输出 (VR180/VR360 SBS/TB 等距柱状视频 + 深度 + PLY)
```

---

## 功能特性

### VR 全景转换 (v2.0)

| 功能 | 说明 |
|------|------|
| 输入投影 | 等距柱状 360°/180°、鱼眼 (等距/等立体角/正交/体视/FTheta) |
| 输出投影 | 自动匹配输入角度 (360°→360°, 180°/鱼眼→180°) |
| 立体布局 | SBS 左右 / TB 上下 |
| 瞳距 IPD | 50–80mm，默认 63mm |
| 立体强度 | 0.2–2.5× |
| 输出分辨率 | 4096×4096 (180°) / 4096×2048 (360°) / 8K / 自定义 |
| 面覆盖策略 | 360°: 6面cubemap (轴间90°) · 180°: 4轴最优半球 (倾斜54.74°, 轴间70.5°) |
| 重叠预测 | 112° FOV (FOV_SCALE=1.5)，预测极限56°，确保无缝覆盖 |
| 拼接缝平滑 | 可选，中心权重 smoothstep 衰减 (floor=0.3 防空洞) |
| 渲染器 | HiGS 推理渲染 (推荐) / 标准 gsplat 光栅化 |
| 渲染面分辨率 | 自适应: 180°→eye_w, 360°→eye_w/2, 最低2048, 对齐256 |
| 导出 | PLY 高斯文件 / 深度全景图 |
| 视频编码 | H.264 / H.265 / AV1 (NVENC GPU 优先) |

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

独立窗口交互式 3D 查看器：鼠标拖拽旋转（连续30fps渲染）、滚轮缩放、拖入 PLY 文件加载。

### 导出选项

| 选项 | 说明 |
|------|------|
| 深度图 | 伪彩色可视化（近=暖色，远=冷色）；视频输出为独立 H.264 文件 |
| PLY 高斯 | 逐帧导出编号序列 `{stem}_{00000}.ply`，兼容内置查看器及公开渲染器 |

---

## 性能

测试环境：i9-13900HX + RTX 5070 Ti Laptop 12GB

### SBS 管线 (4K 输入 → 7680×2160 SBS 输出)

| 指标 | 数值 |
|------|------|
| 端到端单帧 | ~0.55s（~1.83 fps，含解码/编码） |
| SHARP 推理 (TRT FP16) | ~512ms |
| GPU 计算利用率 | ~96% |
| VRAM 占用 | ~4.3GB |

### VR 管线 (4K 等距柱状 → 4096×4096 VR180 SBS)

| 指标 | 数值 |
|------|------|
| 单帧 (4面预测+渲染) | ~3.5s |
| 180° 输出 | 5面×2眼渲染 (跳过-Z) |
| 360° 输出 | 6面×2眼渲染 |
| VRAM 占用 | ~6–8GB |

### 优化技术栈

| 优化 | 加速比 | 说明 |
|------|--------|------|
| ORT TensorRT FP16 | 1.74× | DINOv2 双编码器 (418ms → 240ms) |
| torch.compile max-autotune | 1.16× | 持久缓存，热启动 10.8s |
| GPU-direct prepare_input | — | VR面提取结果直传模型，消除PCIe往返 |
| HiGS 场景复用 | — | fp16打包一次，双眼渲染复用 |
| skip_back 优化 | — | 180°输出跳过-Z面 (6→5面) |
| GPU 四元数 (Shepperd) | 884× | 替代 scipy CPU 实现 |
| 解析法特征分解 | 3× | 替代 GPU SVD (省 0.4s/帧) |
| 3 级流水线 | — | 解码预取 / GPU 计算 / 编码重叠 |
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
| 磁盘 | ~15GB（模型权重 + 编译缓存） |

---

## 安装

### 预编译包 (推荐)

从 [GitHub Releases](https://github.com/appleandhacker/sharp3d/releases) 下载分卷压缩包，合并解压后运行 `sharp3d.exe`：

```bat
copy /b sharp3d-v2.0.0-beta-win64.zip.part_* sharp3d-v2.0.0-beta-win64.zip
```

### 从源码安装

```bash
# 1. 创建虚拟环境
python -m venv sharp3d-env
sharp3d-env\Scripts\activate

# 2. 安装 PyTorch (CUDA 13.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 3. 安装 gsplat（从 GitHub Releases 下载预编译 whl）
pip install gsplat-*.whl

# 4. 安装 SHARP 模型（editable）
git clone https://github.com/apple/ml-sharp
pip install -e ml-sharp

# 5. 安装其余依赖
pip install -r requirements.txt
```

完整依赖清单见 [requirements.txt](requirements.txt)。

---

## 使用

### 图形界面

```bat
set PYTHONPATH=<path-to>\sharp3d\src
python -m sharp3d.gui
```

三个标签页：全景转换 (VR)、SBS 立体转换、2.5D 视差动画。另有独立高斯查看器窗口。

首次启动约 90 秒（模型下载 + TensorRT 引擎构建 + torch.compile），后续启动约 15 秒（持久缓存）。模型就绪后进度栏显示各加速方案启用状态（绿色✔/红色✘）。

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
├── predict.py         模型加载 + FP16 + compile + 预热 + 加速状态收集
├── ort_engine.py      ONNX Runtime TensorRT 加速引擎
├── pipeline.py        端到端管线（图片模式）
├── conversion.py      视频转换引擎（时域稳定 + 边缘柔化 + 卡尔曼收敛）
├── temporal.py        时域稳定器（全局/自适应/光流 + 属性 EMA）
├── projection.py      投影工具（cubemap/半球面提取 + 球面映射 + 角度权重）
├── render.py          gsplat 批量 SBS 渲染 + 单视角 + 深度图
├── render_vr.py       VR 立体渲染（cubemap 6面×2眼 + HiGS + 等距柱状组装）
├── unproject.py       NDC → 世界空间高斯反投影 (GPU-direct)
├── quaternion.py      GPU 四元数转换（Shepperd 法）
├── eigendecompose.py  3×3 对称矩阵特征分解（解析法 + SVD）
├── formats.py         6 种立体格式打包
├── video.py           视频读写 + 编码器探测 + 音频混流
├── hdr.py             HDR 检测 + 色调映射 + HDR10 编码
├── calibrate_int8.py  TensorRT INT8 校准表生成
├── cli.py             命令行入口
└── gui/
    ├── main_window.py 主窗口（多进程架构，GPU 崩溃隔离）
    ├── vr_tab.py      全景转换标签页 (VR180/VR360)
    ├── sbs_tab.py     SBS 立体转换标签页
    ├── anim_tab.py    2.5D 视差动画标签页
    ├── gaussian_tab.py 高斯查看器（独立窗口，连续30fps轨道渲染）
    ├── worker.py      GPU 工作进程（VR/SBS/动画管线 + PLY导出）
    ├── widgets.py     自定义控件（进度条/GPU 监控/预览等）
    └── theme.py       系统主题检测 + 红青配色
```

---

## 已知限制

- **HDR 为格式级支持**：SHARP 是 SDR 模型（sRGB 输入），HDR 输入先色调映射为 SDR 再处理。
- **内部分辨率固定 1536×1536**：SHARP 架构约束（SPN 三级金字塔），与输入分辨率无关。
- **高斯数量固定 ~118 万/面**：模型固定输出，不可调节。VR 管线合并多面后总量更大。
- **首次启动较慢**：TensorRT 引擎构建 + torch.compile 约 90 秒，后续有持久缓存（~15 秒）。
- **VR 拼接缝**：多面预测在重叠区可能有轻微颜色/深度不连续，可启用"拼接缝平滑"选项缓解。

---

## 许可

本项目代码供个人学习研究使用。SHARP 模型权重及其源码遵循 [Apple 原始许可协议](https://github.com/apple/ml-sharp/blob/main/LICENSE)。
