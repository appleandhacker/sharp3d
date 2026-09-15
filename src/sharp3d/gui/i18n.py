"""GUI i18n: Chinese source strings → English runtime mapping.

Design:
- Source code keeps Chinese strings (readable, no key indirection).
- ``tr(text)`` maps a Chinese string to English when the active language
  is English; unknown strings pass through unchanged (fails safe).
- Language is persisted in QSettings ("sharp3d/sharp3d/language"); both the
  GUI process and the GPU worker subprocess read the same store, so worker
  status messages are translated too.
- Switching language takes effect after an app restart (UI is built once).
- Default is Chinese to preserve legacy behavior on untouched machines.
"""

from __future__ import annotations

import os

_LANG: str | None = None

# Populated below. Keys are exact Chinese source strings (as they appear
# in the code, including punctuation/spacing).
_ZH2EN: dict[str, str] = {
    # ---- generic ----
    "自动": "Auto",
    "关闭": "Off",
    "保存": "Save",
    "取消": "Cancel",
    "打开": "Open",
    "导出": "Export",
    "出错": "Error",
    "就绪": "Ready",
    "高级": "Advanced",
    "工具": "Tools",
    "操作": "Actions",
    "分辨率": "Resolution",
    "精度": "Precision",
    "编码器": "Encoder",
    "浏览…": "Browse…",
    "加载中…": "Loading…",
    "图片": "Images",
    "输入": "Input",
    "输出": "Output",
    "自动 (读取元数据)": "Auto (read metadata)",
    "自定义": "Custom",
    "自定义宽度": "Custom width",
    "原始分辨率": "Native resolution",
    "跟随源": "Follow source",
    "0 帧": "0 frames",
    "暗色模式": "Dark mode",
    "亮色模式": "Light mode",
    "已取消": "Cancelled",
    "GPU 不可用": "GPU unavailable",
    "保存…": "Save…",
    # ---- file filters ----
    "图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)":
        "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)",
    "视频 (*.mp4 *.mkv *.avi *.mov *.webm);;所有文件 (*)":
        "Videos (*.mp4 *.mkv *.avi *.mov *.webm);;All files (*)",
    "视频 (*.mp4)": "Videos (*.mp4)",
    "Gaussian PLY (*.ply);;所有文件 (*)":
        "Gaussian PLY (*.ply);;All files (*)",
    # ---- main window / viewer ----
    "平面立体转换": "SBS Stereo Conversion",
    "全景转换": "Panorama Conversion",
    "2.5D 视差动画": "2.5D Parallax Animation",
    "高斯查看器": "Gaussian Viewer",
    "高斯查看器 — sharp3d": "Gaussian Viewer — sharp3d",
    "正在加载模型…": "Loading model…",
    "模型就绪": "Model ready",
    "模型就绪 · ": "Model ready · ",
    "模型量化精度：未加载": "Model quantization: not loaded",
    "模型量化精度：{}": "Model quantization: {}",
    "打开 PLY…": "Open PLY…",
    "重置视角": "Reset view",
    "打开高斯 PLY 文件": "Open Gaussian PLY file",
    "拖入 PLY 文件打开\n左键旋转 · 滚轮缩放":
        "Drop a PLY file to open\nLeft-drag to orbit · scroll to zoom",
    "拖入图片或视频\n开始立体转换":
        "Drop an image or video\nto start stereo conversion",
    "拖入文件/文件夹，或点击浏览…": "Drop files/folders here, or click Browse…",
    "  未加载": "  Not loaded",
    "  {:,} 高斯点 · 拖拽旋转 · 滚轮缩放":
        "  {:,} Gaussians · left-drag orbit · scroll zoom",
    # ---- input / output ----
    "输入 / 输出": "Input / Output",
    "选择文件": "Select file",
    "选择目录": "Select folder",
    "选择文件夹（批量转换）": "Select folder (batch conversion)",
    "选择文件夹（批量）": "Select folder (batch)",
    "文件夹中没有找到支持的图片/视频文件":
        "No supported image/video files found in the folder",
    "已识别 {} 个文件（批量模式）": "{} files detected (batch mode)",
    "请先选择输入路径": "Select an input path first",
    "请先选择输出路径": "Select an output path first",
    # ---- stereo params ----
    "立体参数": "Stereo Parameters",
    "瞳距 IPD": "Pupillary distance (IPD)",
    "收敛分位": "Convergence quantile",
    "0 = 自动(50%) · 值越大前景突出越多 · 100=全部突出":
        "0 = auto (50%) · higher values push the foreground out more · 100 = all out",
    "立体强度": "Stereo strength",
    "立体格式": "Stereo format",
    "导出文件的立体打包格式。\nAnaglyph 红青 = 用红青 3D 眼镜观看；Cross Eyed = 斗鸡眼观看法。":
        "Stereo packing of the exported file.\nAnaglyph = view with red/cyan glasses; Cross Eyed = cross-eye viewing.",
    "立体排列": "Stereo layout",
    "SBS (左右)": "SBS (side by side)",
    "TB (上下)": "TB (top-bottom)",
    "SBS：左右眼水平拼接，VR 播放器标准格式。\nTB：上下眼垂直拼接，部分播放器偏好。":
        "SBS: eyes packed horizontally — the standard for VR players.\nTB: eyes packed vertically — preferred by some players.",
    # ---- output settings ----
    "输出设置": "Output Settings",
    "输出分辨率": "Output resolution",
    "源尺寸 (100%)": "Source size (100%)",
    "按百分比缩放输出分辨率（等比，高度自动）。\n模型推理成本固定，但渲染+编码随像素数变化——\n降到 50% 像素数变为 1/4，渲染/编码约快 4 倍。":
        "Scale output resolution by percentage (aspect kept, height auto).\nInference cost is fixed; rendering + encoding scale with pixels —\nat 50% there are 1/4 the pixels, so rendering/encoding is ~4x faster.",
    "单眼输出宽度（像素），高度按源宽高比自动计算":
        "Per-eye output width in pixels; height follows the source aspect ratio",
    " px / 眼": " px / eye",
    "每只眼的等距柱状分辨率。\nSBS 总宽度 = 2 × 每眼宽度；TB 总高度 = 2 × 每眼高度。\n4096×2048 为主流 VR 头显标准。":
        "Equirectangular resolution per eye.\nSBS total width = 2 × eye width; TB total height = 2 × eye height.\n4096×2048 is the mainstream headset standard.",
    "每眼等距柱状宽度。\n高度自动：180° 输出 → 高度=宽度 (1:1)；\n360° 输出 → 高度=宽度/2 (2:1)。":
        "Equirectangular width per eye.\nHeight is automatic: 180° output → height = width (1:1);\n360° output → height = width/2 (2:1).",
    "4096 × 4096 / 眼 (180° 标准)": "4096 × 4096 / eye (180° standard)",
    "4096 × 2048 / 眼 (360°)": "4096 × 2048 / eye (360°)",
    "7680 × 3840 / 眼 (360° 8K)": "7680 × 3840 / eye (360° 8K)",
    "3840 × 3840 / 眼 (180° 轻量)": "3840 × 3840 / eye (180° lite)",
    "输出帧率": "Output frame rate",
    "输出视频的帧率。\n[跟随源] 保持输入视频原帧率；选择固定值会抽帧/补帧。\n29.97/59.94 为 NTSC 标准帧率。":
        "Output frame rate.\n[Follow source] keeps the input frame rate; a fixed value drops/duplicates frames.\n29.97/59.94 are NTSC standard rates.",
    "保留原始音轨": "Keep original audio track",
    "HDR10 输出 (10-bit PQ)": "HDR10 output (10-bit PQ)",
    "将立体渲染封装为 HDR10 格式，在 HDR 设备上正确显示。\n输入为 HDR 时自动开启。注：模型为 SDR，输出动态范围为 SDR 级。":
        "Package the stereo render as HDR10 for correct display on HDR devices.\nEnabled automatically for HDR input. Note: the model is SDR, so the output dynamic range is SDR-grade.",
    "质量 CRF": "Quality (CRF)",
    "H.264/H.265 质量系数（越小质量越高、文件越大）。\n可直接输入任意 0-51 的值；典型范围 16-28。":
        "H.264/H.265 quality factor (lower = better quality, larger file).\nType any value 0-51; typical range is 16-28.",
    "质量系数（越小质量越高、文件越大）。可直接输入任意 0-51 的值，AV1 硬编内部自动映射到 AV1 量化器（如 40→200）。\nAV1 压缩率高，同画质体积比 H.264 小；典型范围 26-40。":
        "Quality factor (lower = better quality, larger file). Type any value 0-51; AV1 hardware encoding maps it internally to the AV1 quantizer (e.g. 40→200).\nAV1 is more efficient: smaller files at the same quality; typical range is 26-40.",
    "VR 视频建议 CRF 18~20（更高质量减少纱窗效应）。可直接输入任意 0-51 的值。\nAV1 硬编内部自动映射到 AV1 量化器（如 20→100）；AV1 压缩率高，同画质体积更小。":
        "CRF 18-20 recommended for VR video (higher quality reduces screen-door effect). Type any value 0-51.\nAV1 hardware encoding maps it internally to the AV1 quantizer (e.g. 20→100); AV1 is more efficient, smaller files at the same quality.",
    "VR 视频推荐 AV1（压缩率最高，画质最好）。\nH.265 兼容性好；H.264 最广泛但文件较大。":
        "AV1 recommended for VR video (best compression and quality).\nH.265 is broadly compatible; H.264 is the most universal but larger.",
    "AV1 默认走 GPU 硬件编码 (NVENC)，不可用时自动回退 CPU":
        "AV1 uses GPU hardware encoding (NVENC) by default; falls back to CPU if unavailable",
    "当前 ffmpeg 不支持任何 AV1 编码器（需要 libsvtav1 / av1_nvenc / libaom-av1 之一）":
        "Current ffmpeg supports no AV1 encoder (needs libsvtav1 / av1_nvenc / libaom-av1)",
    "编码器: {} ({}) · 输出 {}×{}": "Encoder: {} ({}) · output {}×{}",
    "输出 {}×{} 超过NVENC分辨率上限，已回退CPU编码 ({})，CPU占用会较高":
        "Output {}×{} exceeds the NVENC resolution limit; fell back to CPU encoding ({}) — CPU usage will be high",
    "⚠ 输出 {}×{} 超过NVENC 8192px上限，将使用CPU编码（速度较慢）":
        "⚠ Output {}×{} exceeds the NVENC 8192px limit; CPU encoding will be used (slower)",
    # ---- performance / renderer ----
    "性能模式": "Performance mode",
    "画质优先": "Quality first",
    "速度优先": "Speed first",
    "画质优先 (FP16)": "Quality first (FP16)",
    "速度优先 (FP16)": "Speed first (FP16)",
    "画质优先：FP16 TensorRT + 完整 35 patches（几乎无损）\n速度优先：FP16 TensorRT + 精简 21 patches（提速 ~35%，边缘细节略降）\n\n切换后需重新开始转换生效。":
        "Quality first: FP16 TensorRT + full 35 patches (nearly lossless)\nSpeed first: FP16 TensorRT + trimmed 21 patches (~35% faster, slightly less edge detail)\n\nRestart the conversion after switching for it to take effect.",
    "动画场景重建所用管线的精度：\n画质优先：FP16 TensorRT + 35 patches（默认）\n速度优先：FP16 TensorRT + 21 patches\n切换精度后，下次点击「生成动画」会自动重建场景。":
        "Precision of the pipeline used for animation scene reconstruction:\nQuality first: FP16 TensorRT + 35 patches (default)\nSpeed first: FP16 TensorRT + 21 patches\nAfter switching, clicking \"Generate Animation\" rebuilds the scene automatically.",
    "预测间隔": "Prediction interval",
    "每帧预测 (默认)": "Every frame (default)",
    "每 2 帧 (~1.8x)": "Every 2 frames (~1.8x)",
    "每 3 帧 (~2.4x)": "Every 3 frames (~2.4x)",
    "每 4 帧 (~2.8x)": "Every 4 frames (~2.8x)",
    "每 5 帧 (~3.1x)": "Every 5 frames (~3.1x)",
    "视频关键帧几何复用：每 N 帧完整运行一次 SHARP 预测，\n中间帧复用关键帧几何、仅用当前画面刷新颜色。\n场景切换会自动强制重新预测。\n\n快速运动的物体可能有轻微几何滞后（1-2 帧），\n静态/慢速镜头几乎无损。开启深度图/PLY 导出时不生效。":
        "Keyframe geometry reuse for video: SHARP prediction runs fully every N frames;\nin-between frames reuse keyframe geometry, refreshing only colors.\nScene cuts force a fresh prediction automatically.\n\nFast-moving objects may lag slightly in geometry (1-2 frames);\nstatic/slow shots are nearly lossless. Inactive when depth/PLY export is on.",
    "视频关键帧几何复用：每 N 帧对全部 cubemap 面完整运行一次\nSHARP 预测，中间帧复用关键帧几何、仅用当前画面刷新颜色。\n全景每帧需预测 4-6 个面，复用收益比普通视频更大。\n场景切换会自动强制重新预测。":
        "Keyframe geometry reuse for video: a full SHARP prediction over all cubemap\nfaces runs every N frames; in-between frames reuse keyframe geometry,\nrefreshing only colors. Panoramas predict 4-6 faces per frame, so reuse pays\noff more than for regular video. Scene cuts force a fresh prediction.",
    "渲染器": "Renderer",
    "标准光栅化": "Standard rasterization",
    "HiGS 推理渲染": "HiGS inference rendering",
    "HiGS 推理渲染 (推荐)": "HiGS inference rendering (recommended)",
    "标准光栅化：gsplat rasterization()，支持 batch 多视角、深度输出。\nHiGS 推理渲染：fp16 packed + macro-tile fused，速度快 2x+，\n  质量无损 (PSNR>63dB)，但不支持深度图输出。":
        "Standard: gsplat rasterization(); supports batch multi-view and depth output.\nHiGS inference: fp16 packed + macro-tile fused kernels, 2x+ faster,\n  lossless (PSNR>63dB), but no depth output.",
    "HiGS 推理渲染：fp16 packed + macro-tile fused，\n  cubemap 12 面快 1.56x，显存省 300MB，质量无损。\n标准光栅化：gsplat rasterization() batch 模式，\n  支持深度图输出。":
        "HiGS inference: fp16 packed + macro-tile fused kernels;\n  1.56x faster on 12 cubemap faces, saves 300MB VRAM, lossless.\nStandard: gsplat rasterization() batch mode;\n  supports depth output.",
    "分解方法": "Decomposition method",
    "解析法 (快)": "Analytical (fast)",
    "SVD (参考)": "SVD (reference)",
    "深度稳定": "Depth stabilization",
    "全局平滑 (最快, +1ms/帧)": "Global smoothing (fastest, +1 ms/frame)",
    "自适应平滑 (推荐, +2ms/帧)": "Adaptive smoothing (recommended, +2 ms/frame)",
    "光流稳定 (最佳, +50ms/帧)": "Optical-flow smoothing (best, +50 ms/frame)",
    "视频转换时消除帧间抖动（元素左右跳动/闪烁）。\n\n关闭：不做处理，每帧独立。\n全局对齐：收敛平面EMA平滑 + 深度尺度对齐。\n  消除自动收敛逐帧跳动引起的全局水平偏移。\n自适应：同上 + 逐像素置信度加权深度平滑，\n  静态区域更强平滑，运动物体自动保护。\n光流：同上 + RAFT光流warp遮挡感知混合，\n  处理前景/背景独立运动，质量最佳但较慢(+50ms/帧)。":
        "Removes inter-frame jitter (element jumpiness / flicker) in video conversion.\n\nOff: no processing; frames are independent.\nGlobal: convergence-plane EMA smoothing + depth scale alignment.\n  Removes global horizontal shifts caused by per-frame auto-convergence.\nAdaptive: as above + per-pixel confidence-weighted depth smoothing;\n  stronger smoothing in static regions, moving objects protected.\nOptical flow: as above + RAFT flow warp with occlusion-aware blending;\n  handles independent foreground/background motion; best quality, slower (+50 ms/frame).",
    "深度边缘柔化 (减少边缘拉丝)": "Depth edge softening (less edge smearing)",
    "对深度图边缘做保边平滑，减少立体渲染时\n物体边界处的拉伸/彩色条纹伪影。开销约3ms/帧。":
        "Edge-preserving smoothing on depth boundaries reduces stretching / color\nfringing at object edges during stereo rendering. Costs ~3 ms/frame.",
    "同时输出深度图": "Also output depth map",
    "同时输出深度全景图": "Also output depth panorama",
    "导出 PLY 高斯文件": "Export PLY Gaussian file",
    "导出PLY高斯文件": "Export PLY Gaussian file",
    "拼接缝平滑（中心权重衰减）": "Seam smoothing (center-weight falloff)",
    "对重叠区高斯施加角度衰减以柔化拼接缝。\n关闭时保留全部高斯，可能有轻微接缝但无空洞。":
        "Applies angular falloff to Gaussians in overlap regions to soften seams.\nOff keeps all Gaussians: a faint seam is possible but no holes.",
    "镜头焦距": "Lens focal length",
    "拍摄镜头的 35mm 等效焦距 (mm)。自动：视频按 40mm 等效估算，照片读取 EXIF（无则 30mm）。长焦素材请填真实焦距（如 135）——长焦画面被按广角解释，会导致场景被拉远、立体感扁平。可直接输入任意 8-800 的数值；仅对转换生效，切换后重新开始转换。":
        "35mm-equivalent focal length of the shooting lens (mm). Auto: videos estimated at 40mm equivalent, photos read EXIF (else 30mm). For telephoto footage enter the real focal length (e.g. 135) — telephoto frames interpreted as wide-angle push the scene away and flatten depth. Type any value 8-800; applies when a conversion starts, so restart the conversion after changing.",
    "拍摄镜头的 35mm 等效焦距 (mm)。自动：视频按 40mm 等效估算，照片读取 EXIF（无则 30mm）。长焦素材请填真实焦距（如 135），否则场景会被拉远、立体感扁平。可直接输入任意 8-800 的数值；切换后下次生成动画自动按新焦距重建场景。":
        "35mm-equivalent focal length of the shooting lens (mm). Auto: videos estimated at 40mm equivalent, photos read EXIF (else 30mm). For telephoto footage enter the real focal length (e.g. 135), otherwise the scene is pushed away and depth looks flat. Type any value 8-800; the next \"Generate Animation\" rebuilds the scene at the new focal length.",
    # ---- VR projection ----
    "投影设置": "Projection Settings",
    "输入投影": "Input projection",
    "等距柱状投影": "Equirectangular",
    "鱼眼": "Fisheye",
    "输入视频的投影类型。\n\n等距柱状 (Equirectangular)：标准全景格式，支持360°/180°。\n鱼眼：圆形视场嵌在矩形画面中，输出固定180°。":
        "Projection type of the input video.\n\nEquirectangular: standard panorama format, supports 360°/180°.\nFisheye: circular field of view in a rectangular frame; output is fixed 180°.",
    "覆盖范围": "Coverage",
    "投影模型": "Projection model",
    "等距 (r=fθ)": "Equidistant (r=fθ)",
    "等立体角 (r=2f·sin(θ/2))": "Equi-solid angle (r=2f·sin(θ/2))",
    "正交 (r=f·sinθ)": "Orthographic (r=f·sinθ)",
    "体视 (r=2f·tan(θ/2))": "Stereographic (r=2f·tan(θ/2))",
    "FTheta 多项式 (通用拟合)": "FTheta polynomial (general fit)",
    "三次项系数": "Cubic coefficient",
    "五次项系数": "Quintic coefficient",
    "七次项系数": "Septic coefficient",
    "输出投影自动匹配输入 · 360°输入→360°输出 · 180°/鱼眼→180°输出":
        "Output projection matches input · 360° in → 360° out · 180°/fisheye in → 180° out",
    "全景场景建议 0.6~1.0x 强度": "0.6-1.0x strength recommended for panoramic scenes",
    # ---- conversion progress ----
    "转换进度": "Conversion progress",
    "开始转换": "Start conversion",
    "正在初始化…": "Initializing…",
    "正在取消…": "Cancelling…",
    "转换已取消": "Conversion cancelled",
    "转换失败: {}": "Conversion failed: {}",
    "转换完成 → {}": "Conversion complete → {}",
    "文件 {}/{} · {}": "File {}/{} · {}",
    "批量完成 · 共 {} 个文件": "Batch finished · {} files in total",
    "批量转换完成 · {} 个文件": "Batch conversion complete · {} files",
    "帧 {}/{} · {:.2f} fps · 已用 {} · 剩余 {}":
        "Frame {}/{} · {:.2f} fps · elapsed {} · remaining {}",
    "[{}/{}] 帧 {}/{} · {:.2f} fps · 本文件剩余 {} · 批量剩余 ~{}":
        "[{}/{}] Frame {}/{} · {:.2f} fps · file remaining {} · batch ~{}",
    "完成 · {} 帧 · {:.2f} fps · {}": "Done · {} frames · {:.2f} fps · {}",
    "检测到 HDR 输入，已启用 HDR10 输出": "HDR input detected; HDR10 output enabled",
    "性能模式已切换，正在重建管线…": "Performance mode changed; rebuilding pipeline…",
    "断点续转：从第 {} 帧（{:.3f}s）开始，剩余 {} 帧":
        "Resuming from frame {} ({:.3f}s); {} frames remaining",
    "预加载失败: {}": "Preload failed: {}",
    "准备失败: {}": "Preparation failed: {}",
    "预览渲染失败: {}": "Preview render failed: {}",
    "已重建 3D 场景 · {}×{}": "3D scene rebuilt · {}×{}",
    "{} 失败: {}": "{} failed: {}",
    "错误: {}": "Error: {}",
    "请先加载一张图片": "Load an image first",
    "请先加载 PLY 文件": "Load a PLY file first",
    "请先生成动画（帧已渲染后才可导出）": "Generate the animation first (frames must be rendered before export)",
    "PLY导出失败(不影响转换): {}": "PLY export failed (conversion unaffected): {}",
    "PLY 序列导出中: {}/": "Exporting PLY sequence: {}/",
    "PLY 加载失败: {}": "PLY load failed: {}",
    "已加载 PLY · {:,} 高斯点": "Loaded PLY · {:,} Gaussians",
    "渲染失败: {}": "Render failed: {}",
    "子进程启动失败: {}": "Failed to start subprocess: {}",
    "当前 ffmpeg 不支持任何 AV1 编码器（需要 libsvtav1 / av1_nvenc / libaom-av1 之一）": "Current ffmpeg supports no AV1 encoder (needs libsvtav1 / av1_nvenc / libaom-av1)",
    # ---- animation tab ----
    "动画设置": "Animation Settings",
    "相机轨迹": "Camera trajectory",
    "横扫": "Swipe",
    "摇晃": "Shake",
    "环绕": "Orbit",
    "前推": "Push-in",
    "视差幅度": "Parallax amplitude",
    "缩放幅度": "Zoom amplitude",
    "帧数": "Frames",
    "循环": "Loops",
    "播放帧率": "Playback fps",
    "生成动画": "Generate Animation",
    "渲染进度": "Render progress",
    "渲染中 {}/{}": "Rendering {}/{}",
    "{} 帧": "{} frames",
    "完成 · {} 帧": "Done · {} frames",
    "导出动画": "Export Animation",
    "导出视频…": "Export Video…",
    "正在导出 {} …": "Exporting {} …",
    "正在导出视频…": "Exporting video…",
    "已导出 → {}": "Exported → {}",
    "动画已导出 → {}": "Animation exported → {}",
    "动画渲染失败: {}": "Animation render failed: {}",
    "动画导出失败: {}": "Animation export failed: {}",
    "正在重建 3D 场景…": "Reconstructing 3D scene…",
    "场景就绪 · 点击「生成动画」": "Scene ready · click \"Generate Animation\"",
    "场景重建完成 · {:,} 高斯": "Scene reconstructed · {:,} Gaussians",
    "场景重建完成（{}）· 开始渲染动画":
        "Scene reconstructed ({}) · rendering animation",
    "正在按新焦距/精度重建 3D 场景…":
        "Reconstructing 3D scene with new focal length / precision…",
    "渲染输出宽度（高度按源图宽高比）。\n4K 时 240 帧约需 5.7 GB 内存用于暂存帧，\n导出前请留意内存余量。":
        "Render output width (height follows the source aspect ratio).\nAt 4K, 240 frames need ~5.7 GB RAM for frame staging —\ncheck available memory before exporting.",
    "输入图片": "Input image",
    "语言已切换": "Language changed",
    "界面语言将在下次启动 sharp3d 时生效。": "The UI language will take effect the next time sharp3d starts.",
    '加载模型权重': 'Loading model weights',
    '下载模型权重 (首次需联网，约2.4GB)': 'Downloading model weights (first run needs internet, ~2.4GB)',
    '构建网络结构': 'Building network',
    '载入权重': 'Loading weights',
    '整模型 TRT 引擎（就绪）': 'Full-model TRT engine (ready)',
    '整模型 TRT 引擎（首次构建，约 10-30 分钟）': 'Full-model TRT engine (first build, ~10-30 min)',
    'TensorRT 引擎（就绪）': 'TensorRT engines (ready)',
    'TensorRT 引擎（首次构建，需数分钟）': 'TensorRT engines (first build, takes a few minutes)',
    '整模型 TRT 就绪（跳过 torch.compile）': 'Full-model TRT ready (skipping torch.compile)',
    'TensorRT 引擎（2/2）': 'TensorRT engines (2/2)',
    'TensorRT 引擎': 'TensorRT engines',
    '预热整模型 TRT 引擎': 'Warming up full-model TRT engine',
    '跳过编译（打包模式）': 'Skipping compile (packaged mode)',
    '编译预测器': 'Compiling predictor',
    '跳过编译（回退 eager）': 'Skipping compile (fallback to eager)',
    '预热推理': 'Warming up inference',
    '编译渲染内核': 'Compiling render kernels',
    "Anaglyph 红青": "Anaglyph red/cyan",
    '画质优先：完整 6 面 cubemap 高精度渲染。\n速度优先：降低 cubemap 面分辨率后上采样，提速约 40%。': 'Quality first: full 6-face cubemap rendering at high precision.\nSpeed first: renders cubemap faces at reduced resolution then upscales, ~40% faster.',
    '批量完成 · 共 {} 个文件 · 跳过 {} 个失败': 'Batch finished · {} files total · {} skipped',
    '批量转换完成 · {} 个文件 · 跳过 {} 个失败': 'Batch conversion finished · {} files · {} skipped',
    '批量转换完成 · {} 个文件 · 跳过 {} 个失败: {}': 'Batch conversion finished · {} files · {} skipped: {}',
    '文件出错已跳过 · {}': 'File failed, skipped · {}',
}


def _detect_system_language() -> str:
    """Best-effort OS UI language detection (Windows + POSIX)."""
    env = (os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES")
           or os.environ.get("LANG") or "")
    if env[:2].lower() == "zh":
        return "zh"
    if env[:2].lower() == "en":
        return "en"
    try:
        import ctypes

        windll = ctypes.windll.kernel32
        lang_id = windll.GetUserDefaultUILanguage() & 0xFF
        return "zh" if lang_id == 0x04 else "en"
    except Exception:
        return "zh"


def current_language() -> str:
    """Active language: 'zh' or 'en' (cached after first read)."""
    global _LANG
    if _LANG is None:
        from PySide6.QtCore import QSettings

        saved = QSettings("sharp3d", "sharp3d").value("language", "")
        if saved in ("zh", "en"):
            _LANG = saved
        else:
            _LANG = _detect_system_language()
    return _LANG


def set_language(lang: str) -> None:
    """Persist language choice; effective after app restart."""
    from PySide6.QtCore import QSettings

    assert lang in ("zh", "en")
    QSettings("sharp3d", "sharp3d").setValue("language", lang)


def tr(text: str) -> str:
    """Translate a Chinese source string to the active language."""
    if _LANG == "en" or (_LANG is None and current_language() == "en"):
        return _ZH2EN.get(text, text)
    return text
