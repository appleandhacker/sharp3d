"""SBS conversion tab — compact two-column layout (no preview pane).

Layout: IO card on top, stereo params (left) + output/advanced (right),
progress card at the bottom.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
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

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
IMG_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
VID_FILTER = "视频 (*.mp4 *.mkv *.avi *.mov *.webm);;所有文件 (*)"


def _fmt_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS (omit hours if zero)."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


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
        io_card = SectionCard(c, "输入 / 输出")
        self._input = FileField(c, "输入", file_filter=f"{IMG_FILTER};;{VID_FILTER}")
        self._output = FileField(c, "输出", save=True)
        self._input.path_selected.connect(self._on_input)
        io_card.add_widget(self._input)
        # Folder browse for batch mode
        folder_row = QHBoxLayout()
        btn_folder = QPushButton("选择文件夹（批量）")
        btn_folder.clicked.connect(self._on_browse_folder)
        folder_row.addWidget(btn_folder)
        folder_row.addStretch(1)
        io_card.add_layout(folder_row)
        io_card.add_widget(self._output)
        root.addWidget(io_card)

        # ---- middle: two-column params ------------------------------------
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # Left column: stereo parameters
        left = QVBoxLayout()
        left.setSpacing(12)

        stereo_card = SectionCard(c, "立体参数")
        self._s_ipd = StereoSlider(c, "瞳距 IPD", 50, 80, 63, fmt="{:.0f}",
                                   unit="mm", stereo=True)
        self._s_conv = StereoSlider(c, "收敛深度", 0, 20, 0, fmt="{:.1f}",
                                    unit="m")
        self._s_strength = StereoSlider(c, "立体强度", 0.2, 2.5, 1.0,
                                        fmt="{:.2f}", unit="x")
        conv_hint = QLabel("0 = 自动(25%前景突出) · 值越大前景突出越多")
        conv_hint.setProperty("cssClass", "hint")
        stereo_card.add_widget(self._s_ipd)
        stereo_card.add_widget(self._s_conv)
        stereo_card.add_widget(self._s_strength)
        stereo_card.add_widget(conv_hint)
        left.addWidget(stereo_card)
        left.addStretch(1)

        left_w = QWidget()
        left_w.setLayout(left)
        middle.addWidget(left_w, 1)

        # Right column: output settings + advanced
        right = QVBoxLayout()
        right.setSpacing(12)

        enc_card = SectionCard(c, "输出设置")
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("立体格式"))
        self._format = QComboBox()
        for _key, label in FORMATS:
            self._format.addItem(label)
        self._format.setToolTip(
            "导出文件的立体打包格式。\n"
            "Anaglyph 红青 = 用红青 3D 眼镜观看；Cross Eyed = 斗鸡眼观看法。"
        )
        fmt_row.addWidget(self._format, 1)
        enc_card.add_layout(fmt_row)

        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel("编码器"))
        self._codec = QComboBox()
        self._codec.addItems(["AV1", "H.264", "H.265"])
        self._codec.setToolTip("AV1 默认走 GPU 硬件编码 (NVENC)，不可用时自动回退 CPU")
        codec_row.addWidget(self._codec, 1)
        enc_card.add_layout(codec_row)

        crf_row = QHBoxLayout()
        crf_row.addWidget(QLabel("质量 CRF"))
        self._crf = QComboBox()
        self._crf.addItems(["16", "18", "20", "23", "26", "28"])
        self._crf.setCurrentText("26")
        crf_row.addWidget(self._crf, 1)
        enc_card.add_layout(crf_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("输出帧率"))
        self._fps = QComboBox()
        self._fps.addItems(["跟随源", "24", "29.97", "30", "59.94", "60"])
        self._fps.setToolTip(
            '输出视频的帧率。\n'
            '[跟随源] 保持输入视频原帧率；选择固定值会抽帧/补帧。\n'
            '29.97/59.94 为 NTSC 标准帧率。'
        )
        fps_row.addWidget(self._fps, 1)
        enc_card.add_layout(fps_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("输出分辨率"))
        self._res_scale = QComboBox()
        self._res_scale.addItems(["源尺寸 (100%)", "75%", "50%", "25%", "自定义宽度"])
        self._res_scale.setToolTip(
            "按百分比缩放输出分辨率（等比，高度自动）。\n"
            "模型推理成本固定，但渲染+编码随像素数变化——\n"
            "降到 50% 像素数变为 1/4，渲染/编码约快 4 倍。"
        )
        self._res_scale.currentIndexChanged.connect(self._on_res_changed)
        res_row.addWidget(self._res_scale, 1)
        enc_card.add_layout(res_row)

        self._res_custom_row = QHBoxLayout()
        self._res_custom_row.addWidget(QLabel("自定义宽度"))
        self._res_width = QSpinBox()
        self._res_width.setRange(64, 15360)
        self._res_width.setSingleStep(2)
        self._res_width.setSuffix(" px")
        self._res_width.setValue(1920)
        self._res_width.setToolTip("单眼输出宽度（像素），高度按源宽高比自动计算")
        self._res_custom_row.addWidget(self._res_width, 1)
        self._res_custom_row_enabled = False
        enc_card.add_layout(self._res_custom_row)
        self._set_res_custom_visible(False)

        self._chk_audio = QCheckBox("保留原始音轨")
        self._chk_audio.setChecked(True)
        enc_card.add_widget(self._chk_audio)

        self._chk_hdr = QCheckBox("HDR10 输出 (10-bit PQ)")
        self._chk_hdr.setToolTip(
            "将立体渲染封装为 HDR10 格式，在 HDR 设备上正确显示。\n"
            "输入为 HDR 时自动开启。注：模型为 SDR，输出动态范围为 SDR 级。"
        )
        self._chk_hdr.toggled.connect(self._on_hdr_toggled)
        enc_card.add_widget(self._chk_hdr)
        right.addWidget(enc_card)

        adv_card = SectionCard(c, "高级")
        perf_row = QHBoxLayout()
        perf_row.addWidget(QLabel("性能模式"))
        self._perf_mode = QComboBox()
        self._perf_mode.addItems(["画质优先", "速度优先"])
        self._perf_mode.setToolTip(
            "画质优先：FP16 TensorRT + 完整 35 patches（几乎无损）\n"
            "速度优先：FP16 TensorRT + 精简 21 patches（提速 ~35%，边缘细节略降）\n\n"
            "切换后需重新开始转换生效。"
        )
        perf_row.addWidget(self._perf_mode, 1)
        adv_card.add_layout(perf_row)

        dec_row = QHBoxLayout()
        dec_row.addWidget(QLabel("分解方法"))
        self._decompose = QComboBox()
        self._decompose.addItems(["解析法 (快)", "SVD (参考)"])
        dec_row.addWidget(self._decompose, 1)
        adv_card.add_layout(dec_row)
        self._chk_depth = QCheckBox("同时输出深度图")
        self._chk_ply = QCheckBox("导出 PLY 高斯文件")
        adv_card.add_widget(self._chk_depth)
        adv_card.add_widget(self._chk_ply)
        right.addWidget(adv_card)
        right.addStretch(1)

        right_w = QWidget()
        right_w.setLayout(right)
        middle.addWidget(right_w, 1)

        root.addLayout(middle, 1)

        # ---- progress card ------------------------------------------------
        prog_card = SectionCard(c, "转换进度")
        self._progress = AnimatedProgressBar(c)
        prog_card.add_widget(self._progress)

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

        # ---- engine wiring ------------------------------------------------
        self.request_convert.connect(engine.convert)
        engine.model_loading.connect(self._on_model_loading)
        engine.model_load_progress.connect(self._on_model_load_progress)
        engine.model_ready.connect(self._on_model_ready)
        engine.convert_progress.connect(self._on_convert_progress)
        engine.convert_done.connect(self._on_convert_done)
        engine.error.connect(self._on_error)

    # ------------------------------------------------------------------
    def _on_hdr_toggled(self, checked: bool) -> None:
        if checked and self._codec.currentText() == "H.264":
            self._codec.setCurrentText("H.265")

    def _on_model_loading(self) -> None:
        self._progress.set_value(0.0)
        self._progress.set_busy(True)
        self._prog_label.setText("正在初始化…")

    def _on_model_load_progress(self, stage: str, pct: int) -> None:
        self._progress.set_busy(False)
        self._progress.set_value(pct / 100.0)
        self._prog_label.setText(f"{stage}… {pct}%")

    def _on_model_ready(self) -> None:
        self._progress.set_value(1.0)
        self._progress.set_busy(False)
        self._prog_label.setText("模型就绪")

    def _on_res_changed(self, idx: int) -> None:
        # Last item ("自定义宽度") reveals the custom width spinbox.
        custom = (self._res_scale.currentText() == "自定义宽度")
        self._set_res_custom_visible(custom)

    def _set_res_custom_visible(self, visible: bool) -> None:
        for i in range(self._res_custom_row.count()):
            item = self._res_custom_row.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)

    def _on_browse_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "选择文件夹（批量转换）")
        if path:
            self._input.set_path(path)

    def _on_input(self, path: str) -> None:
        p = Path(path)
        # Folder input: scan for supported files
        if p.is_dir():
            exts = VIDEO_EXTS | {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
            if not files:
                self.status_message.emit("文件夹中没有找到支持的图片/视频文件")
                return
            self._batch_files = [str(f) for f in files]
            self._batch_idx = 0
            self._is_video = files[0].suffix.lower() in VIDEO_EXTS
            self._output.set_path(str(p))
            self.status_message.emit(f"已识别 {len(files)} 个文件（批量模式）")
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
                    self.status_message.emit("检测到 HDR 输入，已启用 HDR10 输出")
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

        # Batch mode: folder was selected
        p = Path(inp)
        if p.is_dir() and not self._batch_files:
            self._on_input(inp)
        if self._batch_files:
            self._batch_idx = 0
            self._start_batch_item()
            return

        # Single file mode
        out = self._output.path()
        if not out:
            self.status_message.emit("请先选择输出路径")
            return
        self._converting = True
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.set_busy(False)
        self._progress.set_value(0.0)
        self.request_convert.emit(self._build_opts(inp, out))

    def _build_opts(self, inp: str, out: str) -> dict:
        """Build conversion options dict for a single file."""
        codec_map = {"H.264": "h264", "H.265": "h265", "AV1": "av1"}
        fps_text = self._fps.currentText()
        out_fps = None if fps_text == "跟随源" else float(fps_text)
        scale_map = {"源尺寸 (100%)": 1.0, "75%": 0.75, "50%": 0.5, "25%": 0.25}
        res_text = self._res_scale.currentText()
        if res_text == "自定义宽度":
            out_width = self._res_width.value()
            out_scale = 1.0
        else:
            out_width = None
            out_scale = scale_map.get(res_text, 1.0)
        return {
            "input": inp,
            "output": out,
            "format": FORMATS[self._format.currentIndex()][0],
            "ipd_mm": self._s_ipd.value(),
            "convergence": self._s_conv.value(),
            "strength": self._s_strength.value(),
            "codec": codec_map[self._codec.currentText()],
            "crf": int(self._crf.currentText()),
            "audio": self._chk_audio.isChecked(),
            "decompose": "analytical" if self._decompose.currentIndex() == 0 else "svd",
            "depth": self._chk_depth.isChecked(),
            "ply": self._chk_ply.isChecked(),
            "hdr_output": self._chk_hdr.isChecked(),
            "perf_mode": "quality" if self._perf_mode.currentIndex() == 0 else "speed",
            "out_fps": out_fps,
            "out_scale": out_scale,
            "out_width": out_width,
        }

    def _start_batch_item(self) -> None:
        """Start converting the current item in the batch queue."""
        if self._batch_idx >= len(self._batch_files):
            self._converting = False
            self._btn_start.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._progress.set_value(1.0)
            self._prog_label.setText(
                f"批量完成 · 共 {len(self._batch_files)} 个文件")
            self.status_message.emit(
                f"批量转换完成 · {len(self._batch_files)} 个文件")
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
            f"文件 {self._batch_idx + 1}/{len(self._batch_files)} · {p.name}")
        self.request_convert.emit(self._build_opts(inp, out))

    def _on_cancel(self) -> None:
        self._engine.cancel()
        self.status_message.emit("正在取消…")

    def _on_convert_progress(self, frame: int, total: int, fps: float,
                             elapsed: float) -> None:
        self._last_fps = fps
        self._progress.set_value(frame / total if total else 0.0)
        remain = (total - frame) / fps if fps > 0 else 0.0
        batch_prefix = ""
        if self._batch_files:
            batch_prefix = (f"[{self._batch_idx + 1}/{len(self._batch_files)}] ")
        self._prog_label.setText(
            f"{batch_prefix}帧 {frame}/{total} · {fps:.2f} fps · "
            f"已用 {_fmt_hms(elapsed)} · 剩余 {_fmt_hms(remain)}"
        )

    def _on_convert_done(self, result: dict) -> None:
        if result.get("cancelled"):
            self._converting = False
            self._btn_start.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._progress.set_busy(False)
            self._prog_label.setText("已取消")
            self.status_message.emit("转换已取消")
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
            n = len(self._batch_files)
            self._prog_label.setText(f"批量完成 · 共 {n} 个文件")
            self.status_message.emit(f"批量转换完成 · {n} 个文件")
            self._batch_files = []
            return

        # Single file done
        self._converting = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.set_value(1.0)
        self._progress.set_busy(False)
        self._prog_label.setText(
            f"完成 · {result['n_frames']} 帧 · {result['fps']:.2f} fps · {result['output']}"
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
        """Re-apply colors after an OS theme switch."""
        self._progress._colors = c
        for s in (self._s_ipd, self._s_conv, self._s_strength):
            s.set_colors(c)
