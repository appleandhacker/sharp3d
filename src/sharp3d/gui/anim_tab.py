"""2.5D parallax animation tab.

Renders a camera trajectory around the reconstructed 3D scene (SHARP's four
trajectory types) and exports as video. No inline preview — use the Gaussian
Viewer tab for interactive 3D preview.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .theme import Colors, ThemeManager
from .widgets import AnimatedProgressBar, FileField, SectionCard, StereoSlider
from .worker import EngineProcess
from .i18n import tr

IMG_FILTER = tr("图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)")
ANIM_RESOLUTIONS = [
    # (label, render width; -1 = source width)
    ("720p  (1280)", 1280),
    ("1080p (1920)", 1920),
    ("1440p (2560)", 2560),
    ("4K    (3840)", 3840),
    (tr("原始分辨率"), -1),
]
ANIM_DEFAULT_RES = 3  # 4K

TRAJECTORIES = [
    ("swipe", tr("横扫")),
    ("shake", tr("摇晃")),
    ("rotate", tr("环绕")),
    ("rotate_forward", tr("前推")),
]


class TrajectoryPicker(QWidget):
    """Segmented control for the four SHARP trajectory types."""

    changed = Signal(str)

    def __init__(self, colors: Colors, parent=None) -> None:
        super().__init__(parent)
        self._colors = colors
        self._buttons = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        from PySide6.QtWidgets import QButtonGroup
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, label in TRAJECTORIES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("cssClass", "traj")
            btn.clicked.connect(lambda _=False, k=key: self._emit(k))
            group.addButton(btn)
            layout.addWidget(btn, 1)
            self._buttons[key] = btn
        self._buttons["rotate_forward"].setChecked(True)
        self._current = "rotate_forward"
        self._apply_style()

    def _emit(self, key: str) -> None:
        self._current = key
        self._apply_style()
        self.changed.emit(key)

    def value(self) -> str:
        return self._current

    def _apply_style(self) -> None:
        c = self._colors
        for key, btn in self._buttons.items():
            if btn.isChecked():
                btn.setStyleSheet(
                    f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                    f"stop:0 {c.red}, stop:1 {c.cyan});"
                    f"color:#0e1013; border:none; font-weight:700;"
                    f"border-radius:8px; padding:7px 4px;"
                )
            else:
                btn.setStyleSheet(
                    f"background:{c.raised}; color:{c.text_muted};"
                    f"border:1px solid {c.border}; border-radius:8px; padding:7px 4px;"
                )

    def set_colors(self, colors: Colors) -> None:
        self._colors = colors
        self._apply_style()


class AnimTab(QWidget):
    """2.5D parallax animation workspace (no inline preview)."""

    status_message = Signal(str)

    request_prepare = Signal(str, int, str, object)
    request_render_anim = Signal(dict)
    request_export_anim = Signal(dict)

    def __init__(self, theme: ThemeManager, engine: EngineProcess, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._engine = engine
        self._prepared = False
        self._awaiting_prepare = False
        self._has_anim = False
        self._prep_mode = None
        self._prep_focal = None
        self._pending_render = False
        self._input_path = ""
        self._rendering = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(12)
        c = theme.colors

        # ---- IO card ------------------------------------------------------
        io_card = SectionCard(c, tr("输入图片"))
        self._input = FileField(c, tr("图片"), file_filter=IMG_FILTER)
        self._input.path_selected.connect(self._on_input)
        io_card.add_widget(self._input)
        root.addWidget(io_card)

        # ---- params + export in two columns ------------------------------
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # left column: trajectory + animation settings
        left = QVBoxLayout()
        left.setSpacing(12)

        traj_card = SectionCard(c, tr("相机轨迹"))
        self._picker = TrajectoryPicker(c)
        traj_card.add_widget(self._picker)
        self._s_disparity = StereoSlider(c, tr("视差幅度"), 0.02, 0.25, 0.08, fmt="{:.2f}")
        self._s_zoom = StereoSlider(c, tr("缩放幅度"), 0.0, 0.4, 0.15, fmt="{:.2f}")
        traj_card.add_widget(self._s_disparity)
        traj_card.add_widget(self._s_zoom)
        left.addWidget(traj_card)

        steps_card = SectionCard(c, tr("动画设置"))
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel(tr("分辨率")))
        self._res = QComboBox()
        for label, _w in ANIM_RESOLUTIONS:
            self._res.addItem(label)
        self._res.setCurrentIndex(ANIM_DEFAULT_RES)
        self._res.setToolTip(
            tr("渲染输出宽度（高度按源图宽高比）。\n4K 时 240 帧约需 5.7 GB 内存用于暂存帧，\n导出前请留意内存余量。")
        )
        res_row.addWidget(self._res, 1)
        steps_card.add_layout(res_row)

        prec_row = QHBoxLayout()
        prec_row.addWidget(QLabel(tr("精度")))
        self._prec = QComboBox()
        self._prec.addItems([tr("画质优先 (FP16)"), tr("速度优先 (FP16)"), tr("FP32 高精度")])
        self._prec.setToolTip(
            tr("动画场景重建所用管线的精度：\n画质优先：FP16 TensorRT + 35 patches（默认）\n速度优先：FP16 TensorRT + 21 patches\nFP32 高精度：纯 torch 单精度，理论质量上限最高，\n  速度最慢，首次使用需下载 FP32 权重（约 2.4GB）\n切换精度后，下次点击「生成动画」会自动重建场景。")
        )
        prec_row.addWidget(self._prec, 1)
        steps_card.add_layout(prec_row)

        focal_row = QHBoxLayout()
        focal_row.addWidget(QLabel(tr("镜头焦距")))
        self._focal = QComboBox()
        self._focal.setEditable(True)
        self._focal.addItems(
            [tr("自动 (读取元数据)"), "24", "28", "35", "50", "85", "135", "200"])
        self._focal.setCurrentIndex(0)
        self._focal.setValidator(QIntValidator(8, 800, self._focal))
        self._focal.setToolTip(
            tr("拍摄镜头的 35mm 等效焦距 (mm)。自动：视频按 40mm 等效估算，照片读取 EXIF（无则 30mm）。长焦素材请填真实焦距（如 135），否则场景会被拉远、立体感扁平。可直接输入任意 8-800 的数值；切换后下次生成动画自动按新焦距重建场景。")
        )
        focal_row.addWidget(self._focal, 1)
        steps_card.add_layout(focal_row)
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel(tr("帧数")))
        self._steps = QSpinBox()
        self._steps.setRange(10, 240)
        self._steps.setValue(60)
        steps_row.addWidget(self._steps, 1)
        steps_card.add_layout(steps_row)

        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel(tr("循环")))
        self._repeats = QSpinBox()
        self._repeats.setRange(1, 5)
        self._repeats.setValue(1)
        rep_row.addWidget(self._repeats, 1)
        steps_card.add_layout(rep_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel(tr("播放帧率")))
        self._fps = QComboBox()
        self._fps.addItems(["24", "30", "60"])
        self._fps.setCurrentText("30")
        fps_row.addWidget(self._fps, 1)
        steps_card.add_layout(fps_row)
        left.addWidget(steps_card)

        middle.addLayout(left, 2)

        # right column: actions + export
        right = QVBoxLayout()
        right.setSpacing(12)

        action_card = SectionCard(c, tr("操作"))
        self._btn_render = QPushButton(tr("生成动画"))
        self._btn_render.clicked.connect(self._on_render)
        action_card.add_widget(self._btn_render)
        self._frame_info = QLabel(tr("0 帧"))
        self._frame_info.setProperty("cssClass", "mono")
        action_card.add_widget(self._frame_info)
        right.addWidget(action_card)

        export_card = SectionCard(c, tr("导出"))
        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel(tr("编码器")))
        self._codec = QComboBox()
        self._codec.addItems(["AV1", "H.264", "H.265"])
        codec_row.addWidget(self._codec, 1)
        export_card.add_layout(codec_row)
        self._btn_export = QPushButton(tr("导出视频…"))
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        export_card.add_widget(self._btn_export)
        right.addWidget(export_card)

        middle.addLayout(right, 2)
        root.addLayout(middle, 1)

        # ---- progress -----------------------------------------------------
        prog_card = SectionCard(c, tr("渲染进度"))
        self._progress = AnimatedProgressBar(c)
        prog_card.add_widget(self._progress)
        self._prog_label = QLabel(tr("就绪"))
        self._prog_label.setProperty("cssClass", "mono")
        prog_card.add_widget(self._prog_label)
        root.addWidget(prog_card)

        # ---- engine wiring ------------------------------------------------
        self.request_prepare.connect(engine.prepare)
        self.request_render_anim.connect(engine.render_anim)
        self.request_export_anim.connect(engine.export_anim)

        engine.prepared.connect(self._on_prepared)
        engine.anim_progress.connect(self._on_anim_progress)
        engine.anim_done.connect(self._on_anim_done)
        engine.anim_exported.connect(self._on_exported)
        engine.error.connect(self._on_error)

    # ------------------------------------------------------------------
    def _on_input(self, path: str) -> None:
        self._prepared = False
        self._has_anim = False
        self._btn_export.setEnabled(False)
        self._awaiting_prepare = True
        self._input_path = path
        self._prep_mode = ("quality", "speed", "fp32")[self._prec.currentIndex()]
        self._prep_focal = self._parse_focal()
        self._prog_label.setText(tr("正在重建 3D 场景…"))
        self.request_prepare.emit(path, 0, self._prep_mode, self._prep_focal)

    def _on_prepared(self, info: dict) -> None:
        if not self._awaiting_prepare:
            return
        self._awaiting_prepare = False
        self._prepared = True
        if self._pending_render:
            self._pending_render = False
            self.status_message.emit(tr("场景重建完成（{}）· 开始渲染动画").format(self._prec.currentText()))
            self._on_render()
            return
        self._prog_label.setText(tr("场景就绪 · 点击「生成动画」"))
        self.status_message.emit(tr("场景重建完成 · {:,} 高斯").format(info['n_gaussians']))

    def _on_render(self) -> None:
        if not self._prepared or self._rendering:
            return
        mode = ("quality", "speed", "fp32")[self._prec.currentIndex()]
        focal = self._parse_focal()
        if (mode != self._prep_mode or focal != self._prep_focal) and self._input_path:
            # Rebuild the scene at the newly selected precision, then render
            # automatically once preparation finishes. _awaiting_prepare MUST
            # be set: _on_prepared() gates on it, and without it the callback
            # returns early — leaving the UI stuck. Do NOT set _rendering
            # here: _on_render() (called from _on_prepared) guards on it and
            # would silently refuse to start the actual render.
            self._pending_render = True
            self._has_anim = False
            self._awaiting_prepare = True
            self._btn_render.setEnabled(False)
            self._btn_export.setEnabled(False)  # frames are now stale-precision
            self._prep_mode = mode
            self._prep_focal = focal
            self._prog_label.setText(tr("正在按新焦距/精度重建 3D 场景…"))
            self.request_prepare.emit(self._input_path, 0, mode, focal)
            return
        self._rendering = True
        self._has_anim = False
        self._btn_export.setEnabled(False)
        self._btn_render.setEnabled(False)
        self._progress.set_value(0.0)
        opts = {
            "type": self._picker.value(),
            "max_disparity": self._s_disparity.value(),
            "max_zoom": self._s_zoom.value(),
            "num_steps": self._steps.value(),
            "num_repeats": self._repeats.value(),
            "preview_width": ANIM_RESOLUTIONS[self._res.currentIndex()][1],
        }
        self.request_render_anim.emit(opts)

    def _on_anim_progress(self, i: int, total: int) -> None:
        self._progress.set_value(i / total if total else 0.0)
        self._prog_label.setText(tr("渲染中 {}/{}").format(i, total))

    def _on_anim_done(self, result: dict) -> None:
        self._rendering = False
        self._btn_render.setEnabled(True)
        self._progress.set_value(1.0)
        n = result["n_frames"]
        self._frame_info.setText(tr("{} 帧").format(n))
        self._prog_label.setText(tr("完成 · {} 帧").format(n))
        if n > 0:
            self._has_anim = True
            self._btn_export.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._rendering = False
        self._awaiting_prepare = False
        self._pending_render = False  # a failed rebuild must not auto-render
        self._btn_render.setEnabled(True)
        if self._has_anim:
            self._btn_export.setEnabled(True)
        self._prog_label.setText(tr("出错"))
        self.status_message.emit(msg)

    # ------------------------------------------------------------------
    def _parse_focal(self) -> float | None:
        """Focal override: None = auto, else clamped 35mm-equivalent mm."""
        text = self._focal.currentText().strip()
        if not text or text.startswith(tr("自动")):
            return None
        try:
            return min(800.0, max(8.0, float(text)))
        except ValueError:
            return None

    def _on_export(self) -> None:
        if not self._has_anim:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, tr("导出动画"), str(Path.home() / "parallax.mp4"),
            tr("视频 (*.mp4)"),
        )
        if not path:
            return
        codec_map = {"H.264": "h264", "H.265": "h265", "AV1": "av1"}
        codec = codec_map[self._codec.currentText()]
        fps = int(self._fps.currentText())
        self.status_message.emit(tr("正在导出 {} …").format(path))
        self._prog_label.setText(tr("正在导出视频…"))
        self._btn_export.setEnabled(False)
        # Frames stay worker-side (up to GBs at 4K) — only the export
        # parameters cross the IPC boundary.
        self.request_export_anim.emit({"path": path, "codec": codec, "fps": fps})

    def _on_exported(self, path: str) -> None:
        self._btn_export.setEnabled(True)
        self._prog_label.setText(tr("已导出 → {}").format(path)[:80])
        self.status_message.emit(tr("动画已导出 → {}").format(path))

    # ------------------------------------------------------------------
    def apply_theme(self, c: Colors) -> None:
        self._progress._colors = c
        self._picker.set_colors(c)
        for s in (self._s_disparity, self._s_zoom):
            s.set_colors(c)
