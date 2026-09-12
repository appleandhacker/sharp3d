"""SBS conversion tab — compact two-column layout (no preview pane).

Layout: IO card on top, stereo params (left) + output/advanced (right),
progress card at the bottom.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .theme import Colors, ThemeManager
from .widgets import (
    AnimatedProgressBar,
    FileField,
    SectionCard,
    StereoSlider,
)
from .worker import EngineProcess

from ..formats import FORMATS
from .i18n import tr

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
IMG_FILTER = tr("图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)")
VID_FILTER = tr("视频 (*.mp4 *.mkv *.avi *.mov *.webm);;所有文件 (*)")


def _fmt_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS (omit hours if zero)."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def precision_text(status: list) -> str:
    """Derive the active model precision from accel_status.

    INT8 was removed (2026-09-10): ORT TRT EP never engaged it for this ViT
    (implicit quantization fell back to FP16 — pixel-identical output) and
    explicit QDQ quantization degraded quality to 25dB. Reporting "INT8"
    was a lie.
    """
    names = {name for name, ok in status if ok}
    if any("TensorRT" in n for n in names):
        return "FP16 (TensorRT)"
    if any("FP16" in n for n in names):
        return "FP16 (PyTorch)"
    return "FP32 (PyTorch)"


class SbsTab(QWidget):
    """Main 2D→3D SBS conversion workspace."""

    status_message = Signal(str)
    request_convert = Signal(dict)

    def __init__(self, theme: ThemeManager, engine: EngineProcess, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._engine = engine
        self._is_video = False
        self._n_frames = 1
        self._converting = False
        self._last_fps = 0.0
        self._batch_files: list[str] = []
        self._batch_idx = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(12)

        c = theme.colors

        # ---- IO card ------------------------------------------------------
        io_card = SectionCard(c, tr("输入 / 输出"))
        self._input = FileField(c, tr("输入"), file_filter=f"{IMG_FILTER};;{VID_FILTER}",
                                allow_folder=True)
        self._output = FileField(c, tr("输出"), save=True)
        self._input.path_selected.connect(self._on_input)
        io_card.add_widget(self._input)
        io_card.add_widget(self._output)
        root.addWidget(io_card)

        # ---- middle: two-column params ------------------------------------
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # Left column: stereo parameters
        left = QVBoxLayout()
        left.setSpacing(12)

        stereo_card = SectionCard(c, tr("立体参数"))
        self._s_ipd = StereoSlider(c, tr("瞳距 IPD"), 50, 80, 63, fmt="{:.0f}",
                                   unit="mm", stereo=True)
        self._s_conv = StereoSlider(c, tr("收敛分位"), 0, 100, 0, fmt="{:.0f}",
                                    unit="%")
        self._s_strength = StereoSlider(c, tr("立体强度"), 0.2, 2.5, 1.0,
                                        fmt="{:.2f}", unit="x")
        conv_hint = QLabel(tr("0 = 自动(50%) · 值越大前景突出越多 · 100=全部突出"))
        conv_hint.setProperty("cssClass", "hint")
        stereo_card.add_widget(self._s_ipd)
        stereo_card.add_widget(self._s_conv)
        stereo_card.add_widget(conv_hint)
        stereo_card.add_widget(self._s_strength)
        left.addWidget(stereo_card)

        # Right column: output settings
        right = QVBoxLayout()
        right.setSpacing(12)

        enc_card = SectionCard(c, tr("输出设置"))
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel(tr("立体格式")))
        self._format = QComboBox()
        for _key, label in FORMATS:
            self._format.addItem(tr(label))
        self._format.setToolTip(
            tr("导出文件的立体打包格式。\nAnaglyph 红青 = 用红青 3D 眼镜观看；Cross Eyed = 斗鸡眼观看法。")
        )
        fmt_row.addWidget(self._format, 1)
        enc_card.add_layout(fmt_row)

        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel(tr("编码器")))
        self._codec = QComboBox()
        self._codec.addItems(["AV1", "H.264", "H.265"])
        self._codec.setToolTip(tr("AV1 默认走 GPU 硬件编码 (NVENC)，不可用时自动回退 CPU"))
        codec_row.addWidget(self._codec, 1)
        enc_card.add_layout(codec_row)

        crf_row = QHBoxLayout()
        crf_row.addWidget(QLabel(tr("质量 CRF")))
        self._crf = QComboBox()
        self._crf.setEditable(True)  # 任意 0-51 可输入，预设仅作快捷项
        self._crf.addItems(["16", "18", "20", "23", "26", "28"])
        self._crf.setValidator(QIntValidator(0, 51, self._crf))
        self._crf.setToolTip(
            tr("H.264/H.265 质量系数（越小质量越高、文件越大）。\n可直接输入任意 0-51 的值；典型范围 16-28。")
        )
        self._crf.setCurrentText("26")
        crf_row.addWidget(self._crf, 1)
        enc_card.add_layout(crf_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel(tr("输出帧率")))
        self._fps = QComboBox()
        self._fps.addItems([tr("跟随源"), "24", "29.97", "30", "59.94", "60"])
        self._fps.setToolTip(
            tr("输出视频的帧率。\n[跟随源] 保持输入视频原帧率；选择固定值会抽帧/补帧。\n29.97/59.94 为 NTSC 标准帧率。")
        )
        fps_row.addWidget(self._fps, 1)
        enc_card.add_layout(fps_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel(tr("输出分辨率")))
        self._res_scale = QComboBox()
        self._res_scale.addItems([tr("源尺寸 (100%)"), "75%", "50%", "25%", tr("自定义宽度")])
        self._res_scale.setToolTip(
            tr("按百分比缩放输出分辨率（等比，高度自动）。\n模型推理成本固定，但渲染+编码随像素数变化——\n降到 50% 像素数变为 1/4，渲染/编码约快 4 倍。")
        )
        self._res_scale.currentIndexChanged.connect(self._on_res_changed)
        res_row.addWidget(self._res_scale, 1)
        enc_card.add_layout(res_row)

        self._res_custom_row = QHBoxLayout()
        self._res_custom_row.addWidget(QLabel(tr("自定义宽度")))
        self._res_width = QSpinBox()
        self._res_width.setRange(64, 15360)
        self._res_width.setSingleStep(2)
        self._res_width.setSuffix(" px")
        self._res_width.setValue(1920)
        self._res_width.setToolTip(tr("单眼输出宽度（像素），高度按源宽高比自动计算"))
        self._res_custom_row.addWidget(self._res_width, 1)
        self._res_custom_row_enabled = False
        enc_card.add_layout(self._res_custom_row)
        self._set_res_custom_visible(False)

        self._chk_audio = QCheckBox(tr("保留原始音轨"))
        self._chk_audio.setChecked(True)
        enc_card.add_widget(self._chk_audio)

        self._chk_hdr = QCheckBox(tr("HDR10 输出 (10-bit PQ)"))
        self._chk_hdr.setToolTip(
            tr("将立体渲染封装为 HDR10 格式，在 HDR 设备上正确显示。\n输入为 HDR 时自动开启。注：模型为 SDR，输出动态范围为 SDR 级。")
        )
        self._chk_hdr.toggled.connect(self._on_hdr_toggled)
        enc_card.add_widget(self._chk_hdr)
        # 输出设置放左列（立体参数下方），高级区域放右列
        left.addWidget(enc_card)

        adv_card = SectionCard(c, tr("高级"))
        perf_row = QHBoxLayout()
        perf_row.addWidget(QLabel(tr("性能模式")))
        self._perf_mode = QComboBox()
        self._perf_mode.addItems([tr("画质优先"), tr("速度优先"), tr("FP32 高精度")])
        self._perf_mode.setToolTip(
            tr("画质优先：FP16 TensorRT + 完整 35 patches（几乎无损）\n速度优先：FP16 TensorRT + 精简 21 patches（提速 ~35%，边缘细节略降）\nFP32 高精度：纯 torch 单精度管线（无 TensorRT），理论质量上限最高；\n  速度最慢，显存约 2.8GB，首次使用需 FP32 权重 sharp_fp32.pt\n\n切换后需重新开始转换生效。")
        )
        perf_row.addWidget(self._perf_mode, 1)
        adv_card.add_layout(perf_row)

        focal_row = QHBoxLayout()
        focal_row.addWidget(QLabel(tr("镜头焦距")))
        self._focal = QComboBox()
        self._focal.setEditable(True)
        self._focal.addItems(
            [tr("自动 (读取元数据)"), "24", "28", "35", "50", "85", "135", "200"])
        self._focal.setCurrentIndex(0)
        self._focal.setValidator(QIntValidator(8, 800, self._focal))
        self._focal.setToolTip(
            tr("拍摄镜头的 35mm 等效焦距 (mm)。自动：视频按 40mm 等效估算，照片读取 EXIF（无则 30mm）。长焦素材请填真实焦距（如 135）——长焦画面被按广角解释，会导致场景被拉远、立体感扁平。可直接输入任意 8-800 的数值；仅对转换生效，切换后重新开始转换。")
        )
        focal_row.addWidget(self._focal, 1)
        adv_card.add_layout(focal_row)

        kf_row = QHBoxLayout()
        kf_row.addWidget(QLabel(tr("预测间隔")))
        self._kf_interval = QComboBox()
        self._kf_interval.addItems([
            tr("每帧预测 (默认)"),
            tr("每 2 帧 (~1.8x)"),
            tr("每 3 帧 (~2.4x)"),
            tr("每 4 帧 (~2.8x)"),
            tr("每 5 帧 (~3.1x)"),
        ])
        # 保持「每帧预测」为默认：kf 复用带来的 1-2 帧几何滞后需要用户
        # 自行取舍（基准数据见 tests/bench.py，kf2 实测 +52% 帧率）。
        self._kf_interval.setToolTip(
            tr("视频关键帧几何复用：每 N 帧完整运行一次 SHARP 预测，\n中间帧复用关键帧几何、仅用当前画面刷新颜色。\n场景切换会自动强制重新预测。\n\n快速运动的物体可能有轻微几何滞后（1-2 帧），\n静态/慢速镜头几乎无损。开启深度图/PLY 导出时不生效。")
        )
        kf_row.addWidget(self._kf_interval, 1)
        adv_card.add_layout(kf_row)

        renderer_row = QHBoxLayout()
        renderer_row.addWidget(QLabel(tr("渲染器")))
        self._renderer = QComboBox()
        self._renderer.addItems([tr("标准光栅化"), tr("HiGS 推理渲染")])
        self._renderer.setToolTip(
            tr("标准光栅化：gsplat rasterization()，支持 batch 多视角、深度输出。\nHiGS 推理渲染：fp16 packed + macro-tile fused，速度快 2x+，\n  质量无损 (PSNR>63dB)，但不支持深度图输出。")
        )
        renderer_row.addWidget(self._renderer, 1)
        adv_card.add_layout(renderer_row)
        # 默认：HiGS（实测 1.42→1.94fps，质量与标准光栅化无差异 42.76/42.77dB）
        self._renderer.setCurrentIndex(1)

        dec_row = QHBoxLayout()
        dec_row.addWidget(QLabel(tr("分解方法")))
        self._decompose = QComboBox()
        self._decompose.addItems([tr("解析法 (快)"), tr("SVD (参考)")])
        dec_row.addWidget(self._decompose, 1)
        adv_card.add_layout(dec_row)

        stab_row = QHBoxLayout()
        stab_row.addWidget(QLabel(tr("深度稳定")))
        self._stabilize = QComboBox()
        self._stabilize.addItems([
            tr("关闭"),
            tr("全局平滑 (最快, +1ms/帧)"),
            tr("自适应平滑 (推荐, +2ms/帧)"),
            tr("光流稳定 (最佳, +50ms/帧)"),
        ])
        self._stabilize.setToolTip(
            tr("视频转换时消除帧间抖动（元素左右跳动/闪烁）。\n\n关闭：不做处理，每帧独立。\n全局对齐：收敛平面EMA平滑 + 深度尺度对齐。\n  消除自动收敛逐帧跳动引起的全局水平偏移。\n自适应：同上 + 逐像素置信度加权深度平滑，\n  静态区域更强平滑，运动物体自动保护。\n光流：同上 + RAFT光流warp遮挡感知混合，\n  处理前景/背景独立运动，质量最佳但较慢(+50ms/帧)。")
        )
        self._stabilize.setCurrentIndex(0)  # 默认：关闭（用户实测时域平滑引入闪烁）
        stab_row.addWidget(self._stabilize, 1)
        adv_card.add_layout(stab_row)

        self._chk_depth = QCheckBox(tr("同时输出深度图"))
        self._chk_ply = QCheckBox(tr("导出 PLY 高斯文件"))
        self._chk_edge = QCheckBox(tr("深度边缘柔化 (减少边缘拉丝)"))
        self._chk_edge.setToolTip(
            tr("对深度图边缘做保边平滑，减少立体渲染时\n物体边界处的拉伸/彩色条纹伪影。开销约3ms/帧。")
        )
        adv_card.add_widget(self._chk_depth)
        adv_card.add_widget(self._chk_ply)
        adv_card.add_widget(self._chk_edge)
        right.addWidget(adv_card)

        left_w = QWidget()
        left_w.setLayout(left)
        middle.addWidget(left_w, 1)

        right.addStretch(1)
        right_w = QWidget()
        right_w.setLayout(right)
        middle.addWidget(right_w, 1)

        root.addLayout(middle, 1)

        # ---- progress card ------------------------------------------------
        prog_card = SectionCard(c, tr("转换进度"))
        prog_bar_row = QHBoxLayout()
        self._progress = AnimatedProgressBar(c)
        prog_bar_row.addWidget(self._progress, 1)
        self._pct_label = QLabel("0%")
        self._pct_label.setProperty("cssClass", "mono")
        self._pct_label.setFixedWidth(40)
        self._pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prog_bar_row.addWidget(self._pct_label)
        prog_card.add_layout(prog_bar_row)

        prog_row = QHBoxLayout()
        self._prog_label = QLabel(tr("就绪"))
        self._prog_label.setProperty("cssClass", "mono")
        prog_row.addWidget(self._prog_label, 1)
        self._btn_cancel = QPushButton(tr("取消"))
        self._btn_cancel.setProperty("cssClass", "danger")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_start = QPushButton(tr("开始转换"))
        self._btn_start.setProperty("cssClass", "primary")
        self._btn_start.clicked.connect(self._on_start)
        prog_row.addWidget(self._btn_cancel)
        prog_row.addWidget(self._btn_start)
        prog_card.add_layout(prog_row)
        root.addWidget(prog_card)

        # ---- engine wiring ------------------------------------------------
        self.request_convert.connect(engine.convert)
        engine.model_loading.connect(self._on_model_loading)
        engine.model_load_progress.connect(self._on_model_load_progress)
        engine.model_ready.connect(self._on_model_ready)
        engine.model_accel.connect(self._on_model_accel)
        engine.convert_progress.connect(self._on_convert_progress)
        engine.convert_done.connect(self._on_convert_done)
        engine.error.connect(self._on_error)

    # ------------------------------------------------------------------
    def _on_hdr_toggled(self, checked: bool) -> None:
        if checked and self._codec.currentText() == "H.264":
            self._codec.setCurrentText("H.265")

    def _on_model_loading(self) -> None:
        self._progress.set_busy(False)
        self._progress.set_value(0.03)
        self._pct_label.setText("3%")
        self._prog_label.setText(tr("正在初始化…"))

    def _on_model_load_progress(self, stage: str, pct: int) -> None:
        self._progress.set_busy(False)
        self._progress.set_value(pct / 100.0)
        self._pct_label.setText(f"{pct}%")
        self._prog_label.setText(f"{stage}… {pct}%")

    def _on_model_ready(self) -> None:
        self._progress.set_value(1.0)
        self._progress.set_busy(False)
        self._pct_label.setText("100%")
        self._prog_label.setText(tr("模型就绪"))

    def _on_model_accel(self, status: list) -> None:
        parts = []
        for name, ok in status:
            if ok:
                parts.append(f'{name}<b style="color:#4caf50;">✔</b>')
            else:
                parts.append(f'{name}<b style="color:#f44336;">✘</b>')
        self._prog_label.setTextFormat(Qt.RichText)
        self._prog_label.setText(tr("模型就绪 · ") + " · ".join(parts))
        self._prog_label.setToolTip("\n".join(
            f"{'✔' if ok else '✘'} {name}" for name, ok in status))

    def _on_res_changed(self, idx: int) -> None:
        # Last item ("自定义宽度") reveals the custom width spinbox.
        custom = (self._res_scale.currentText() == tr("自定义宽度"))
        self._set_res_custom_visible(custom)

    def _set_res_custom_visible(self, visible: bool) -> None:
        for i in range(self._res_custom_row.count()):
            item = self._res_custom_row.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)

    def _on_browse_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, tr("选择文件夹（批量转换）"))
        if path:
            self._input.set_path(path)

    def _on_input(self, path: str) -> None:
        p = Path(path)
        # Folder input: scan for supported files
        if p.is_dir():
            exts = VIDEO_EXTS | {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
            if not files:
                self.status_message.emit(tr("文件夹中没有找到支持的图片/视频文件"))
                return
            self._batch_files = [str(f) for f in files]
            self._batch_idx = 0
            self._is_video = files[0].suffix.lower() in VIDEO_EXTS
            self._output.set_path(str(p))
            self.status_message.emit(tr("已识别 {} 个文件（批量模式）").format(len(files)))
            self._n_frames = 1
            return

        # Single file input
        self._batch_files = []
        self._batch_idx = 0
        self._is_video = p.suffix.lower() in VIDEO_EXTS
        out = p.parent / f"{p.stem}_sbs{('.mp4' if self._is_video else p.suffix)}"
        self._output.set_path(str(out))
        if self._is_video:
            try:
                from sharp3d.hdr import probe_video
                info = probe_video(p)
                self._n_frames = info["n_frames"] or 1
                if info["is_hdr"]:
                    self._chk_hdr.setChecked(True)
                    if self._codec.currentText() == "H.264":
                        self._codec.setCurrentText("H.265")
                    self.status_message.emit(tr("检测到 HDR 输入，已启用 HDR10 输出"))
            except Exception:
                self._n_frames = 1
        else:
            self._n_frames = 1

    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        inp = self._input.path()
        if not inp:
            self.status_message.emit(tr("请先选择输入路径"))
            return

        # Batch mode: folder was selected
        p = Path(inp)
        if p.is_dir() and not self._batch_files:
            self._on_input(inp)
        if self._batch_files:
            # Resume from failed item if _batch_idx > 0 (error recovery),
            # otherwise start fresh.
            if self._batch_idx == 0 or self._batch_idx >= len(self._batch_files):
                self._batch_idx = 0
            self._start_batch_item()
            return

        # Single file mode
        out = self._output.path()
        if not out:
            self.status_message.emit(tr("请先选择输出路径"))
            return
        self._converting = True
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.set_busy(False)
        self._progress.set_value(0.0)
        self.request_convert.emit(self._build_opts(inp, out))

    def _parse_focal(self) -> float | None:
        """Focal override: None = auto, else clamped 35mm-equivalent mm."""
        text = self._focal.currentText().strip()
        if not text or text.startswith(tr("自动")):
            return None
        try:
            return min(800.0, max(8.0, float(text)))
        except ValueError:
            return None

    def _build_opts(self, inp: str, out: str) -> dict:
        """Build conversion options for a single file (typed via ConvertOptions)."""
        from sharp3d.options import ConvertOptions

        codec_map = {"H.264": "h264", "H.265": "h265", "AV1": "av1"}
        fps_text = self._fps.currentText()
        out_fps = None if fps_text == tr("跟随源") else float(fps_text)
        scale_map = {tr("源尺寸 (100%)"): 1.0, "75%": 0.75, "50%": 0.5, "25%": 0.25}
        res_text = self._res_scale.currentText()
        if res_text == tr("自定义宽度"):
            out_width = self._res_width.value()
            out_scale = 1.0
        else:
            out_width = None
            out_scale = scale_map.get(res_text, 1.0)

        opts = ConvertOptions(
            input=inp,
            output=out,
            format=FORMATS[self._format.currentIndex()][0],
            ipd_mm=self._s_ipd.value(),
            convergence=self._s_conv.value() / 100.0,  # 0=auto, else quantile
            strength=self._s_strength.value(),
            codec=codec_map[self._codec.currentText()],
            crf=min(51, max(0, int(self._crf.currentText().strip() or 20))),
            audio=self._chk_audio.isChecked(),
            decompose="analytical" if self._decompose.currentIndex() == 0 else "svd",
            depth=self._chk_depth.isChecked(),
            ply=self._chk_ply.isChecked(),
            edge_soften=self._chk_edge.isChecked(),
            hdr_output=self._chk_hdr.isChecked(),
            perf_mode=("quality", "speed", "fp32")[self._perf_mode.currentIndex()],
            focal_35mm=self._parse_focal(),
            renderer="higs" if self._renderer.currentIndex() == 1 else "standard",
            out_fps=out_fps,
            out_scale=out_scale,
            out_width=out_width,
            temporal_stabilize=["off", "global", "adaptive", "flow"][
                self._stabilize.currentIndex()],
            keyframe_interval=self._kf_interval.currentIndex() + 1,
        )
        return opts.to_dict()

    def _start_batch_item(self) -> None:
        """Start converting the current item in the batch queue."""
        if self._batch_idx >= len(self._batch_files):
            self._converting = False
            self._btn_start.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._progress.set_value(1.0)
            self._prog_label.setText(
                tr("批量完成 · 共 {} 个文件").format(len(self._batch_files)))
            self.status_message.emit(
                tr("批量转换完成 · {} 个文件").format(len(self._batch_files)))
            self._batch_files = []
            return

        inp = self._batch_files[self._batch_idx]
        p = Path(inp)
        is_vid = p.suffix.lower() in VIDEO_EXTS
        out = str(p.parent / f"{p.stem}_sbs{('.mp4' if is_vid else p.suffix)}")

        self._converting = True
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.set_busy(False)
        self._progress.set_value(0.0)
        self._prog_label.setText(
            tr("文件 {}/{} · {}").format(self._batch_idx + 1, len(self._batch_files), p.name))
        self.request_convert.emit(self._build_opts(inp, out))

    def _on_cancel(self) -> None:
        self._engine.cancel()
        self.status_message.emit(tr("正在取消…"))

    def _on_convert_progress(self, frame: int, total: int, fps: float,
                             elapsed: float) -> None:
        # All tabs share one EngineProcess, so conversion events broadcast to
        # every tab. Ignore ones this tab did not start.
        if not self._converting:
            return
        self._last_fps = fps
        file_frac = frame / total if total else 0.0
        self._progress.set_value(file_frac)
        self._pct_label.setText(f"{int(file_frac * 100)}%")
        remain = (total - frame) / fps if fps > 0 else 0.0

        if self._batch_files:
            n_files = len(self._batch_files)
            # Batch-level ETA: current file remaining + remaining files estimate.
            avg_file_time = elapsed / max(file_frac, 0.01)
            batch_remain = remain + (n_files - self._batch_idx - 1) * avg_file_time
            self._prog_label.setText(
                tr("[{}/{}] 帧 {}/{} · {:.2f} fps · 本文件剩余 {} · 批量剩余 ~{}").format(
                    self._batch_idx + 1, n_files, frame, total, fps,
                    _fmt_hms(remain), _fmt_hms(batch_remain))
            )
        else:
            self._prog_label.setText(
                tr("帧 {}/{} · {:.2f} fps · 已用 {} · 剩余 {}").format(
                    frame, total, fps, _fmt_hms(elapsed), _fmt_hms(remain))
            )

    def _on_convert_done(self, result: dict) -> None:
        if not self._converting and not self._batch_files:
            return  # another tab's conversion — ignore
        if result.get("cancelled"):
            self._converting = False
            self._btn_start.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._progress.set_busy(False)
            self._pct_label.setText("0%")
            self._prog_label.setText(tr("已取消"))
            self.status_message.emit(tr("转换已取消"))
            self._batch_files = []
            return

        # Batch mode: auto-start next file
        if self._batch_files:
            self._batch_idx += 1
            if self._batch_idx < len(self._batch_files):
                self._start_batch_item()
                return
            # Batch complete
            self._converting = False
            self._btn_start.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._progress.set_value(1.0)
            self._pct_label.setText("100%")
            n = len(self._batch_files)
            self._prog_label.setText(tr("批量完成 · 共 {} 个文件").format(n))
            self.status_message.emit(tr("批量转换完成 · {} 个文件").format(n))
            self._batch_files = []
            return

        # Single file done
        self._converting = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.set_value(1.0)
        self._progress.set_busy(False)
        self._pct_label.setText("100%")
        self._prog_label.setText(
            tr("完成 · {} 帧 · {:.2f} fps · {}").format(result['n_frames'], result['fps'], result['output'])
        )
        self.status_message.emit(tr("转换完成 → {}").format(result['output']))

    def _on_error(self, msg: str) -> None:
        if not self._converting:
            return  # another tab's conversion — ignore
        self._converting = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.set_busy(False)
        if self._batch_files and self._batch_idx < len(self._batch_files):
            self._prog_label.setText(
                tr("批量 [{}/{}] 出错，点击开始从断点继续").format(
                    self._batch_idx + 1, len(self._batch_files)))
        else:
            self._prog_label.setText(tr("出错"))
        self.status_message.emit(msg)

    # ------------------------------------------------------------------
    def apply_theme(self, c: Colors) -> None:
        """Re-apply colors after an OS theme switch."""
        self._progress._colors = c
        for s in (self._s_ipd, self._s_conv, self._s_strength):
            s.set_colors(c)
