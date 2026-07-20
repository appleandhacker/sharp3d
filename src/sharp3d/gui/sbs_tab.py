"""SBS conversion tab — the main workspace.

Layout: IO card on top, preview (left, stretching) + parameter column (right),
progress card at the bottom. Stereo sliders re-render the cached gaussians
live for responsive IPD/convergence/strength tuning.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .theme import Colors, ThemeManager
from .widgets import (
    AnimatedProgressBar,
    FileField,
    PreviewPane,
    SectionCard,
    StereoSlider,
)
from .worker import EngineProcess

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
IMG_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
VID_FILTER = "视频 (*.mp4 *.mkv *.avi *.mov *.webm);;所有文件 (*)"
PREVIEW_WIDTH = 1280


class SbsTab(QWidget):
    """Main 2D→3D SBS conversion workspace."""

    status_message = Signal(str)

    # Request signals: emitted on the GUI thread. The engine's request
    # methods are non-blocking (they enqueue to the child pipeline process),
    # but going through signals keeps the call path uniform and safe.
    request_prepare = Signal(str, int)
    request_render_preview = Signal(float, float, float, int)
    request_convert = Signal(dict)

    def __init__(self, theme: ThemeManager, engine: EngineProcess, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._engine = engine
        self._prepared = False
        self._awaiting_prepare = False
        self._is_video = False
        self._n_frames = 1
        self._converting = False
        self._last_fps = 0.0

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
        io_card.add_widget(self._output)
        root.addWidget(io_card)

        # ---- middle: preview + params ------------------------------------
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # preview card
        preview_card = SectionCard(c, "立体预览")
        self._preview = PreviewPane(c)
        self._preview.file_dropped.connect(self._input.set_path)
        preview_card.add_widget(self._preview, )

        preview_ctrl = QHBoxLayout()
        self._btn_preview = QPushButton("单帧预览")
        self._btn_preview.clicked.connect(self._do_prepare)
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setEnabled(False)
        self._frame_label = QLabel("帧 —")
        self._frame_label.setProperty("cssClass", "mono")
        self._frame_label.setMinimumWidth(64)
        preview_ctrl.addWidget(self._btn_preview)
        preview_ctrl.addWidget(self._frame_slider, 1)
        preview_ctrl.addWidget(self._frame_label)
        preview_card.add_layout(preview_ctrl)

        middle.addWidget(preview_card, 3)

        # right column of parameter cards
        right = QVBoxLayout()
        right.setSpacing(12)

        stereo_card = SectionCard(c, "立体参数")
        self._s_ipd = StereoSlider(c, "瞳距 IPD", 50, 80, 63, fmt="{:.0f}",
                                   unit="mm", stereo=True)
        self._s_conv = StereoSlider(c, "收敛深度", 0, 20, 0, fmt="{:.1f}",
                                    unit="m")
        self._s_strength = StereoSlider(c, "立体强度", 0.2, 2.5, 1.0,
                                        fmt="{:.2f}", unit="x")
        conv_hint = QLabel("收敛深度 0 = 自动对焦")
        conv_hint.setProperty("cssClass", "hint")
        stereo_card.add_widget(self._s_ipd)
        stereo_card.add_widget(self._s_conv)
        stereo_card.add_widget(self._s_strength)
        stereo_card.add_widget(conv_hint)
        right.addWidget(stereo_card)

        enc_card = SectionCard(c, "视频编码")
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
        self._crf.addItems(["16", "18", "20", "23", "28"])
        self._crf.setCurrentText("18")
        crf_row.addWidget(self._crf, 1)
        enc_card.add_layout(crf_row)

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

        middle.addLayout(right, 2)
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

        # ---- debounced preview re-render ---------------------------------
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._render_preview_now)
        for s in (self._s_ipd, self._s_conv, self._s_strength):
            s.value_changed.connect(lambda _v: self._schedule_preview())

        # Debounce frame-slider prepare: dragging fires many valueChanged
        # events; each prepare costs ~1s, so only run the last one.
        self._frame_prepare_timer = QTimer(self)
        self._frame_prepare_timer.setSingleShot(True)
        self._frame_prepare_timer.setInterval(250)
        self._frame_prepare_timer.timeout.connect(self._do_prepare)

        self._frame_slider.valueChanged.connect(self._on_frame_slider)

        # ---- engine wiring ------------------------------------------------
        # Request signals -> engine request methods (non-blocking; the heavy
        # work runs in the child pipeline process so the GUI stays responsive).
        self.request_prepare.connect(engine.prepare)
        self.request_render_preview.connect(engine.render_preview)
        self.request_convert.connect(engine.convert)

        engine.preview_ready.connect(self._on_preview_ready)
        engine.prepared.connect(self._on_prepared)
        engine.convert_progress.connect(self._on_convert_progress)
        engine.convert_done.connect(self._on_convert_done)
        engine.error.connect(self._on_error)

    # ------------------------------------------------------------------
    def _schedule_preview(self) -> None:
        if self._prepared and not self._converting:
            self._preview_timer.start()

    def _render_preview_now(self) -> None:
        self.request_render_preview.emit(
            self._s_ipd.value(), self._s_conv.value(),
            self._s_strength.value(), PREVIEW_WIDTH,
        )

    def _on_hdr_toggled(self, checked: bool) -> None:
        # HDR10 requires H.265 or AV1; H.264 cannot carry it.
        if checked and self._codec.currentText() == "H.264":
            self._codec.setCurrentText("H.265")

    def _on_input(self, path: str) -> None:
        p = Path(path)
        self._is_video = p.suffix.lower() in VIDEO_EXTS
        # auto-name output
        out = p.parent / f"{p.stem}_sbs{('.mp4' if self._is_video else p.suffix)}"
        self._output.set_path(str(out))
        # frame slider for video
        if self._is_video:
            try:
                from sharp3d.hdr import probe_video
                info = probe_video(p)
                self._n_frames = info["n_frames"] or 1
                # auto-enable HDR10 output for HDR sources
                if info["is_hdr"]:
                    self._chk_hdr.setChecked(True)
                    if self._codec.currentText() == "H.264":
                        self._codec.setCurrentText("H.265")
                    self.status_message.emit("检测到 HDR 输入，已启用 HDR10 输出")
            except Exception:
                self._n_frames = 1
            self._frame_slider.setEnabled(self._n_frames > 1)
            self._frame_slider.setRange(0, max(0, self._n_frames - 1))
            self._frame_slider.setValue(0)
        else:
            self._n_frames = 1
            self._frame_slider.setEnabled(False)
            self._frame_slider.setRange(0, 0)
        self._frame_label.setText(f"帧 0/{max(0, self._n_frames - 1)}")
        self._prepared = False
        self._preview.clear_image()
        self._preview.set_message("正在重建 3D 场景…")
        self._do_prepare()

    def _on_frame_slider(self, idx: int) -> None:
        self._frame_label.setText(f"帧 {idx}/{max(0, self._n_frames - 1)}")
        if self._is_video and not self._converting:
            self._prepared = False
            self._frame_prepare_timer.start()  # debounced prepare

    def _do_prepare(self) -> None:
        path = self._input.path()
        if not path or self._converting:
            return
        idx = self._frame_slider.value() if self._is_video else 0
        self._preview.set_message("正在重建 3D 场景…")
        self._awaiting_prepare = True
        self.request_prepare.emit(path, idx)

    def _on_prepared(self, info: dict) -> None:
        # Both tabs hear engine.prepared; only react to our own request.
        if not self._awaiting_prepare:
            return
        self._awaiting_prepare = False
        self._prepared = True
        self.status_message.emit(
            f"场景重建完成 · {info['n_gaussians']:,} 高斯"
        )
        self._render_preview_now()

    def _on_preview_ready(self, sbs: object) -> None:
        self._preview.set_image(sbs)

    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        inp = self._input.path()
        out = self._output.path()
        if not inp or not out:
            self.status_message.emit("请先选择输入和输出路径")
            return
        self._converting = True
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.set_busy(False)
        self._progress.set_value(0.0)

        codec_map = {"H.264": "h264", "H.265": "h265", "AV1": "av1"}
        opts = {
            "input": inp,
            "output": out,
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
        }
        self.request_convert.emit(opts)

    def _on_cancel(self) -> None:
        # Direct call is intentional: cancel() only sets a flag that the running
        # conversion loop polls. A queued signal would sit behind the busy
        # worker and never arrive in time.
        self._engine.cancel()
        self.status_message.emit("正在取消…")

    def _on_convert_progress(self, frame: int, total: int, fps: float) -> None:
        self._last_fps = fps
        self._progress.set_value(frame / total if total else 0.0)
        elapsed = frame / fps if fps else 0.0
        remain = (total - frame) / fps if fps else 0.0
        self._prog_label.setText(
            f"帧 {frame}/{total} · {fps:.2f} fps · 已用 {elapsed:.0f}s · 剩余 {remain:.0f}s"
        )

    def _on_convert_done(self, result: dict) -> None:
        self._converting = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.set_value(1.0)
        self._progress.set_busy(False)
        if result.get("cancelled"):
            self._prog_label.setText("已取消")
            self.status_message.emit("转换已取消")
        else:
            self._prog_label.setText(
                f"完成 · {result['n_frames']} 帧 · {result['fps']:.2f} fps · {result['output']}"
            )
            self.status_message.emit(f"转换完成 → {result['output']}")

    def _on_error(self, msg: str) -> None:
        self._converting = False
        self._awaiting_prepare = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.set_busy(False)
        self._prog_label.setText("出错")
        self.status_message.emit(msg)

    # ------------------------------------------------------------------
    def apply_theme(self, c: Colors) -> None:
        """Re-apply colors after an OS theme switch."""
        self._preview.set_colors(c)
        self._progress._colors = c
        for s in (self._s_ipd, self._s_conv, self._s_strength):
            s.set_colors(c)
