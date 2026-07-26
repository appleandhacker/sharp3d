"""Panoramic VR conversion tab — equirectangular/fisheye → stereo 3D VR.

Layout: IO card on top, projection params (left) + output settings (right),
progress card at the bottom.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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

VID_FILTER = "视频 (*.mp4 *.mkv *.avi *.mov *.webm);;所有文件 (*)"
IMG_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

# Output resolution presets: (label, width_per_eye, height_per_eye)
RES_PRESETS = [
    ("4096 × 4096 / 眼 (180° 标准)", 4096, 4096),
    ("4096 × 2048 / 眼 (360°)", 4096, 2048),
    ("7680 × 3840 / 眼 (360° 8K)", 7680, 3840),
    ("3840 × 3840 / 眼 (180° 轻量)", 3840, 3840),
    ("自定义", 0, 0),
]


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


class VrTab(QWidget):
    """Panoramic video → stereo 3D VR conversion workspace."""

    status_message = Signal(str)
    request_convert = Signal(dict)

    def __init__(self, theme: ThemeManager, engine: EngineProcess, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._engine = engine
        self._converting = False
        self._n_frames = 1

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(12)

        c = theme.colors

        # ---- IO card ----------------------------------------------------------
        io_card = SectionCard(c, "输入 / 输出")
        self._input = FileField(c, "输入", file_filter=f"{VID_FILTER};;{IMG_FILTER}")
        self._output = FileField(c, "输出", save=True)
        self._input.path_selected.connect(self._on_input)
        io_card.add_widget(self._input)
        io_card.add_widget(self._output)
        root.addWidget(io_card)

        # ---- middle: two-column params ----------------------------------------
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # Left column: projection + stereo
        left = QVBoxLayout()
        left.setSpacing(12)

        proj_card = SectionCard(c, "投影设置")

        in_proj_row = QHBoxLayout()
        in_proj_row.addWidget(QLabel("输入投影"))
        self._in_proj = QComboBox()
        self._in_proj.addItems([
            "自动检测",
            "等距柱状投影",
            "鱼眼",
        ])
        self._in_proj.setToolTip(
            "输入视频的投影类型。\n\n"
            "等距柱状 (Equirectangular)：标准全景格式。\n"
            "鱼眼：圆形视场嵌在矩形画面中。\n"
            "自动检测：根据宽高比和元数据推断。"
        )
        self._in_proj.currentIndexChanged.connect(self._on_in_proj_changed)
        in_proj_row.addWidget(self._in_proj, 1)
        proj_card.add_layout(in_proj_row)

        # Sub-option row (hidden for 自动检测)
        self._in_proj_sub_row = QHBoxLayout()
        self._in_proj_sub_label = QLabel("子类型")
        self._in_proj_sub_row.addWidget(self._in_proj_sub_label)
        self._in_proj_sub = QComboBox()
        self._in_proj_sub.currentIndexChanged.connect(self._on_in_proj_sub_changed)
        self._in_proj_sub_row.addWidget(self._in_proj_sub, 1)
        proj_card.add_layout(self._in_proj_sub_row)
        self._equirect_subs = ["360°", "180°"]
        self._fisheye_subs = ["等距 (r=fθ)", "等立体角 (r=2f·sin(θ/2))",
                              "正交 (r=f·sinθ)", "体视 (r=2f·tan(θ/2))",
                              "FTheta 多项式 (通用拟合)"]

        # FTheta coefficient row (hidden unless FTheta selected)
        self._ftheta_row = QHBoxLayout()
        self._ftheta_row.addWidget(QLabel("r=f·(θ+k₁θ³+k₂θ⁵+k₃θ⁷)"))
        self._ftheta_k1 = QDoubleSpinBox()
        self._ftheta_k1.setRange(-1.0, 1.0)
        self._ftheta_k1.setDecimals(2)
        self._ftheta_k1.setSingleStep(0.01)
        self._ftheta_k1.setPrefix("k₁=")
        self._ftheta_k1.setToolTip("三次项系数")
        self._ftheta_row.addWidget(self._ftheta_k1)
        self._ftheta_k2 = QDoubleSpinBox()
        self._ftheta_k2.setRange(-1.0, 1.0)
        self._ftheta_k2.setDecimals(2)
        self._ftheta_k2.setSingleStep(0.01)
        self._ftheta_k2.setPrefix("k₂=")
        self._ftheta_k2.setToolTip("五次项系数")
        self._ftheta_row.addWidget(self._ftheta_k2)
        self._ftheta_k3 = QDoubleSpinBox()
        self._ftheta_k3.setRange(-1.0, 1.0)
        self._ftheta_k3.setDecimals(2)
        self._ftheta_k3.setSingleStep(0.01)
        self._ftheta_k3.setPrefix("k₃=")
        self._ftheta_k3.setToolTip("七次项系数")
        self._ftheta_row.addWidget(self._ftheta_k3)
        proj_card.add_layout(self._ftheta_row)
        self._set_ftheta_visible(False)
        # Initialize: hide sub-options (自动检测 selected by default)
        self._set_sub_visible(False)

        out_proj_row = QHBoxLayout()
        out_proj_row.addWidget(QLabel("输出投影"))
        self._out_proj = QComboBox()
        self._out_proj.addItems([
            "等距柱状 180°",
            "等距柱状 360°",
        ])
        self._out_proj.setToolTip(
            "输出立体视频的投影格式。\n"
            "180°：仅前半球，VR180 标准，文件更小。\n"
            "360°：完整球面，适用于所有 VR 头显。"
        )
        self._out_proj.currentIndexChanged.connect(self._on_out_proj_changed)
        out_proj_row.addWidget(self._out_proj, 1)
        proj_card.add_layout(out_proj_row)

        fov_hint = QLabel("输出为等距柱状投影 · cubemap 6 面渲染后球面映射")
        fov_hint.setProperty("cssClass", "hint")
        proj_card.add_widget(fov_hint)

        left.addWidget(proj_card)

        stereo_card = SectionCard(c, "立体参数")
        self._s_ipd = StereoSlider(c, "瞳距 IPD", 50, 80, 63, fmt="{:.0f}",
                                   unit="mm", stereo=True)
        self._s_conv = StereoSlider(c, "收敛分位", 0, 100, 0, fmt="{:.0f}",
                                    unit="%")
        self._s_conv.setVisible(False)  # VR管线不使用收敛分位
        self._s_strength = StereoSlider(c, "立体强度", 0.2, 2.5, 1.0,
                                        fmt="{:.2f}", unit="x")
        conv_hint = QLabel("0 = 自动(50%) · 全景场景建议 0.6~1.0x 强度")
        conv_hint.setProperty("cssClass", "hint")
        stereo_card.add_widget(self._s_ipd)
        stereo_card.add_widget(self._s_conv)
        stereo_card.add_widget(conv_hint)
        stereo_card.add_widget(self._s_strength)
        left.addWidget(stereo_card)

        left.addStretch(1)
        left_w = QWidget()
        left_w.setLayout(left)
        middle.addWidget(left_w, 1)

        # Right column: output settings
        right = QVBoxLayout()
        right.setSpacing(12)

        enc_card = SectionCard(c, "输出设置")

        layout_row = QHBoxLayout()
        layout_row.addWidget(QLabel("立体排列"))
        self._stereo_layout = QComboBox()
        self._stereo_layout.addItems(["SBS (左右)", "TB (上下)"])
        self._stereo_layout.setToolTip(
            "SBS：左右眼水平拼接，VR 播放器标准格式。\n"
            "TB：上下眼垂直拼接，部分播放器偏好。"
        )
        layout_row.addWidget(self._stereo_layout, 1)
        enc_card.add_layout(layout_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("输出分辨率"))
        self._res_preset = QComboBox()
        for label, _, _ in RES_PRESETS:
            self._res_preset.addItem(label)
        self._res_preset.setToolTip(
            "每只眼的等距柱状分辨率。\n"
            "SBS 总宽度 = 2 × 每眼宽度；TB 总高度 = 2 × 每眼高度。\n"
            "4096×2048 为主流 VR 头显标准。"
        )
        self._res_preset.currentIndexChanged.connect(self._on_res_changed)
        res_row.addWidget(self._res_preset, 1)
        enc_card.add_layout(res_row)

        self._res_custom_row = QHBoxLayout()
        self._res_custom_row.addWidget(QLabel("自定义宽度"))
        self._res_width = QSpinBox()
        self._res_width.setRange(1024, 15360)
        self._res_width.setSingleStep(256)
        self._res_width.setSuffix(" px / 眼")
        self._res_width.setValue(4096)
        self._res_width.setToolTip(
            "每眼等距柱状宽度。\n"
            "高度自动：180° 输出 → 高度=宽度 (1:1)；\n"
            "360° 输出 → 高度=宽度/2 (2:1)。"
        )
        self._res_custom_row.addWidget(self._res_width, 1)
        enc_card.add_layout(self._res_custom_row)
        self._set_custom_res_visible(False)

        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel("编码器"))
        self._codec = QComboBox()
        self._codec.addItems(["AV1", "H.265", "H.264"])
        self._codec.setToolTip(
            "VR 视频推荐 AV1（压缩率最高，画质最好）。\n"
            "H.265 兼容性好；H.264 最广泛但文件较大。"
        )
        codec_row.addWidget(self._codec, 1)
        enc_card.add_layout(codec_row)

        crf_row = QHBoxLayout()
        crf_row.addWidget(QLabel("质量 CRF"))
        self._crf = QComboBox()
        self._crf.addItems(["16", "18", "20", "23", "26", "28"])
        self._crf.setCurrentText("20")
        self._crf.setToolTip("VR 视频建议 CRF 18~20（更高质量减少纱窗效应）")
        crf_row.addWidget(self._crf, 1)
        enc_card.add_layout(crf_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("输出帧率"))
        self._fps = QComboBox()
        self._fps.addItems(["跟随源", "24", "29.97", "30", "59.94", "60"])
        fps_row.addWidget(self._fps, 1)
        enc_card.add_layout(fps_row)

        self._chk_audio = QCheckBox("保留原始音轨")
        self._chk_audio.setChecked(True)
        enc_card.add_widget(self._chk_audio)

        right.addWidget(enc_card)

        adv_card = SectionCard(c, "高级")

        perf_row = QHBoxLayout()
        perf_row.addWidget(QLabel("性能模式"))
        self._perf_mode = QComboBox()
        self._perf_mode.addItems(["画质优先", "速度优先"])
        self._perf_mode.setToolTip(
            "画质优先：完整 6 面 cubemap 高精度渲染。\n"
            "速度优先：降低 cubemap 面分辨率后上采样，提速约 40%。"
        )
        perf_row.addWidget(self._perf_mode, 1)
        adv_card.add_layout(perf_row)

        renderer_row = QHBoxLayout()
        renderer_row.addWidget(QLabel("渲染器"))
        self._renderer = QComboBox()
        self._renderer.addItems(["HiGS 推理渲染 (推荐)", "标准光栅化"])
        self._renderer.setToolTip(
            "HiGS 推理渲染：fp16 packed + macro-tile fused，\n"
            "  cubemap 12 面快 1.56x，显存省 300MB，质量无损。\n"
            "标准光栅化：gsplat rasterization() batch 模式，\n"
            "  支持深度图输出。"
        )
        renderer_row.addWidget(self._renderer, 1)
        adv_card.add_layout(renderer_row)

        stab_row = QHBoxLayout()
        stab_row.addWidget(QLabel("深度稳定"))
        self._stabilize = QComboBox()
        self._stabilize.addItems([
            "关闭",
            "全局平滑 (推荐)",
            "自适应平滑",
        ])
        self._stabilize.setToolTip(
            "视频帧间深度一致性平滑。\n"
            "全景视频建议至少开启全局平滑。"
        )
        stab_row.addWidget(self._stabilize, 1)
        adv_card.add_layout(stab_row)
        # Hide: VR视频管线尚未实现时域稳定
        for idx in range(stab_row.count()):
            w = stab_row.itemAt(idx).widget()
            if w:
                w.setVisible(False)

        self._chk_depth = QCheckBox("同时输出深度全景图")
        adv_card.add_widget(self._chk_depth)

        self._chk_ply = QCheckBox("导出PLY高斯文件")
        adv_card.add_widget(self._chk_ply)

        right.addWidget(adv_card)
        right.addStretch(1)

        right_w = QWidget()
        right_w.setLayout(right)
        middle.addWidget(right_w, 1)

        root.addLayout(middle, 1)

        # ---- progress card ----------------------------------------------------
        prog_card = SectionCard(c, "转换进度")
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
        self._prog_label = QLabel("就绪")
        self._prog_label.setProperty("cssClass", "mono")
        prog_row.addWidget(self._prog_label, 1)
        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.setProperty("cssClass", "danger")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_start = QPushButton("开始转换")
        self._btn_start.setProperty("cssClass", "primary")
        self._btn_start.clicked.connect(self._on_start)
        prog_row.addWidget(self._btn_cancel)
        prog_row.addWidget(self._btn_start)
        prog_card.add_layout(prog_row)
        root.addWidget(prog_card)

        # ---- engine wiring ----------------------------------------------------
        self.request_convert.connect(engine.convert)
        engine.model_loading.connect(self._on_model_loading)
        engine.model_load_progress.connect(self._on_model_load_progress)
        engine.model_ready.connect(self._on_model_ready)
        engine.convert_progress.connect(self._on_convert_progress)
        engine.convert_done.connect(self._on_convert_done)
        engine.error.connect(self._on_error)

    # ------------------------------------------------------------------
    def _on_out_proj_changed(self, idx: int) -> None:
        """Auto-switch resolution preset when output projection changes."""
        if idx == 0:  # 180° → 1:1
            self._res_preset.setCurrentIndex(0)  # 4096×4096
        else:  # 360° → 2:1
            self._res_preset.setCurrentIndex(1)  # 4096×2048

    def _on_in_proj_changed(self, idx: int) -> None:
        """Show/hide sub-option combo based on input projection type."""
        if idx == 0:  # 自动检测
            self._set_sub_visible(False)
        elif idx == 1:  # 等距柱状投影
            self._in_proj_sub.clear()
            self._in_proj_sub.addItems(self._equirect_subs)
            self._in_proj_sub_label.setText("覆盖范围")
            self._set_sub_visible(True)
        elif idx == 2:  # 鱼眼
            self._in_proj_sub.clear()
            self._in_proj_sub.addItems(self._fisheye_subs)
            self._in_proj_sub_label.setText("投影模型")
            self._set_sub_visible(True)

    def _set_sub_visible(self, visible: bool) -> None:
        for i in range(self._in_proj_sub_row.count()):
            item = self._in_proj_sub_row.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)
        if not visible:
            self._set_ftheta_visible(False)

    def _on_in_proj_sub_changed(self, idx: int) -> None:
        """Show FTheta coefficients only when FTheta model is selected."""
        # FTheta is the last item in fisheye subs (index 4)
        is_ftheta = (self._in_proj.currentIndex() == 2 and idx == 4)
        self._set_ftheta_visible(is_ftheta)

    def _set_ftheta_visible(self, visible: bool) -> None:
        for i in range(self._ftheta_row.count()):
            item = self._ftheta_row.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)

    def _on_res_changed(self, idx: int) -> None:
        custom = (self._res_preset.currentIndex() == len(RES_PRESETS) - 1)
        self._set_custom_res_visible(custom)

    def _set_custom_res_visible(self, visible: bool) -> None:
        for i in range(self._res_custom_row.count()):
            item = self._res_custom_row.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)

    def _on_input(self, path: str) -> None:
        p = Path(path)
        is_video = p.suffix.lower() in VIDEO_EXTS
        suffix = ".mp4" if is_video else p.suffix
        out = p.parent / f"{p.stem}_vr{suffix}"
        self._output.set_path(str(out))
        if is_video:
            try:
                from sharp3d.hdr import probe_video
                info = probe_video(p)
                self._n_frames = info["n_frames"] or 1
            except Exception:
                self._n_frames = 1
        else:
            self._n_frames = 1

    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        inp = self._input.path()
        if not inp:
            self.status_message.emit("请先选择输入路径")
            return
        out = self._output.path()
        if not out:
            self.status_message.emit("请先选择输出路径")
            return

        # NVENC resolution warning
        opts = self._build_opts(inp, out)
        ew, eh = opts["eye_width"], opts["eye_height"]
        layout = opts["stereo_layout"]
        packed_w = ew * 2 if layout == "sbs" else ew
        packed_h = eh if layout == "sbs" else eh * 2
        if packed_w > 8192 or packed_h > 8192:
            self.status_message.emit(
                f"⚠ 输出 {packed_w}×{packed_h} 超过NVENC 8192px上限，"
                f"将使用CPU编码（速度较慢）")

        self._converting = True
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.set_busy(False)
        self._progress.set_value(0.0)
        self.request_convert.emit(opts)

    def _build_opts(self, inp: str, out: str) -> dict:
        """Build VR conversion options."""
        # Resolve output resolution
        idx = self._res_preset.currentIndex()
        if idx < len(RES_PRESETS) - 1:
            _, eye_w, eye_h = RES_PRESETS[idx]
        else:
            eye_w = self._res_width.value()
            # 180° → 1:1, 360° → 2:1
            if self._out_proj.currentIndex() == 0:  # 180°
                eye_h = eye_w
            else:  # 360°
                eye_h = eye_w // 2

        codec_map = {"H.264": "h264", "H.265": "h265", "AV1": "av1"}
        fps_text = self._fps.currentText()
        out_fps = None if fps_text == "跟随源" else float(fps_text)

        # Input projection: main type + sub-option
        main_idx = self._in_proj.currentIndex()
        if main_idx == 0:
            input_projection = "auto"
        elif main_idx == 1:  # 等距柱状
            sub_idx = self._in_proj_sub.currentIndex()
            input_projection = "equirect360" if sub_idx == 0 else "equirect180"
        else:  # 鱼眼
            sub_idx = self._in_proj_sub.currentIndex()
            fisheye_map = {
                0: "fisheye_equidistant",
                1: "fisheye_equisolid",
                2: "fisheye_orthographic",
                3: "fisheye_stereographic",
                4: "fisheye_ftheta",
            }
            input_projection = fisheye_map.get(sub_idx, "fisheye_equidistant")

        # FTheta coefficients (only relevant when fisheye_ftheta selected)
        ftheta_coeffs = None
        if input_projection == "fisheye_ftheta":
            ftheta_coeffs = [
                self._ftheta_k1.value(),
                self._ftheta_k2.value(),
                self._ftheta_k3.value(),
            ]

        out_proj_map = {0: "equirect180", 1: "equirect360"}
        layout_map = {0: "sbs", 1: "tb"}

        return {
            "mode": "vr",
            "input": inp,
            "output": out,
            "input_projection": input_projection,
            "ftheta_coeffs": ftheta_coeffs,
            "output_projection": out_proj_map[self._out_proj.currentIndex()],
            "stereo_layout": layout_map[self._stereo_layout.currentIndex()],
            "eye_width": eye_w,
            "eye_height": eye_h,
            "ipd_mm": self._s_ipd.value(),
            "convergence": self._s_conv.value() / 100.0,
            "strength": self._s_strength.value(),
            "codec": codec_map[self._codec.currentText()],
            "crf": int(self._crf.currentText()),
            "audio": self._chk_audio.isChecked(),
            "perf_mode": "quality" if self._perf_mode.currentIndex() == 0 else "speed",
            "renderer": "standard" if self._renderer.currentIndex() == 1 else "higs",
            "out_fps": out_fps,
            "temporal_stabilize": ["off", "global", "adaptive"][
                self._stabilize.currentIndex()],
            "depth": self._chk_depth.isChecked(),
            "ply": self._chk_ply.isChecked(),
        }

    def _on_cancel(self) -> None:
        self._engine.cancel()
        self.status_message.emit("正在取消…")

    # ------------------------------------------------------------------
    def _on_model_loading(self) -> None:
        self._progress.set_busy(False)
        self._progress.set_value(0.03)
        self._pct_label.setText("3%")
        self._prog_label.setText("正在初始化…")

    def _on_model_load_progress(self, stage: str, pct: int) -> None:
        self._progress.set_busy(False)
        self._progress.set_value(pct / 100.0)
        self._pct_label.setText(f"{pct}%")
        self._prog_label.setText(f"{stage}… {pct}%")

    def _on_model_ready(self) -> None:
        self._progress.set_value(1.0)
        self._progress.set_busy(False)
        self._pct_label.setText("100%")
        self._prog_label.setText("模型就绪")

    def _on_convert_progress(self, frame: int, total: int, fps: float,
                             elapsed: float) -> None:
        file_frac = frame / total if total else 0.0
        self._progress.set_value(file_frac)
        self._pct_label.setText(f"{int(file_frac * 100)}%")
        remain = (total - frame) / fps if fps > 0 else 0.0
        self._prog_label.setText(
            f"帧 {frame}/{total} · {fps:.2f} fps · "
            f"已用 {_fmt_hms(elapsed)} · 剩余 {_fmt_hms(remain)}"
        )

    def _on_convert_done(self, result: dict) -> None:
        if result.get("cancelled"):
            self._converting = False
            self._btn_start.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._progress.set_busy(False)
            self._pct_label.setText("0%")
            self._prog_label.setText("已取消")
            self.status_message.emit("转换已取消")
            return

        self._converting = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.set_value(1.0)
        self._progress.set_busy(False)
        self._pct_label.setText("100%")
        self._prog_label.setText(
            f"完成 · {result['n_frames']} 帧 · "
            f"{result['fps']:.2f} fps · {result['output']}"
        )
        self.status_message.emit(f"转换完成 → {result['output']}")

    def _on_error(self, msg: str) -> None:
        self._converting = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.set_busy(False)
        self._prog_label.setText("出错")
        self.status_message.emit(msg)

    # ------------------------------------------------------------------
    def apply_theme(self, c: Colors) -> None:
        self._progress._colors = c
        for s in (self._s_ipd, self._s_conv, self._s_strength):
            s.set_colors(c)
