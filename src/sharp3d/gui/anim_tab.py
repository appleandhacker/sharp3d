"""2.5D parallax animation tab.

Renders a camera trajectory around the reconstructed 3D scene (SHARP's four
trajectory types) and exports as video. No inline preview — use the Gaussian
Viewer tab for interactive 3D preview.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
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

IMG_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
ANIM_PREVIEW_WIDTH = 960

TRAJECTORIES = [
    ("swipe", "横扫"),
    ("shake", "摇晃"),
    ("rotate", "环绕"),
    ("rotate_forward", "前推"),
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

    request_prepare = Signal(str, int)
    request_render_anim = Signal(dict)
    request_export_anim = Signal(dict)

    def __init__(self, theme: ThemeManager, engine: EngineProcess, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._engine = engine
        self._prepared = False
        self._awaiting_prepare = False
        self._frames: list = []
        self._rendering = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(12)
        c = theme.colors

        # ---- IO card ------------------------------------------------------
        io_card = SectionCard(c, "输入图片")
        self._input = FileField(c, "图片", file_filter=IMG_FILTER)
        self._input.path_selected.connect(self._on_input)
        io_card.add_widget(self._input)
        root.addWidget(io_card)

        # ---- params + export in two columns ------------------------------
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # left column: trajectory + animation settings
        left = QVBoxLayout()
        left.setSpacing(12)

        traj_card = SectionCard(c, "相机轨迹")
        self._picker = TrajectoryPicker(c)
        traj_card.add_widget(self._picker)
        self._s_disparity = StereoSlider(c, "视差幅度", 0.02, 0.25, 0.08, fmt="{:.2f}")
        self._s_zoom = StereoSlider(c, "缩放幅度", 0.0, 0.4, 0.15, fmt="{:.2f}")
        traj_card.add_widget(self._s_disparity)
        traj_card.add_widget(self._s_zoom)
        left.addWidget(traj_card)

        steps_card = SectionCard(c, "动画设置")
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("帧数"))
        self._steps = QSpinBox()
        self._steps.setRange(10, 240)
        self._steps.setValue(60)
        steps_row.addWidget(self._steps, 1)
        steps_card.add_layout(steps_row)

        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel("循环"))
        self._repeats = QSpinBox()
        self._repeats.setRange(1, 5)
        self._repeats.setValue(1)
        rep_row.addWidget(self._repeats, 1)
        steps_card.add_layout(rep_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("播放帧率"))
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

        action_card = SectionCard(c, "操作")
        self._btn_render = QPushButton("生成动画")
        self._btn_render.clicked.connect(self._on_render)
        action_card.add_widget(self._btn_render)
        self._frame_info = QLabel("0 帧")
        self._frame_info.setProperty("cssClass", "mono")
        action_card.add_widget(self._frame_info)
        right.addWidget(action_card)

        export_card = SectionCard(c, "导出")
        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel("编码器"))
        self._codec = QComboBox()
        self._codec.addItems(["AV1", "H.264", "H.265"])
        codec_row.addWidget(self._codec, 1)
        export_card.add_layout(codec_row)
        self._btn_export = QPushButton("导出视频…")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        export_card.add_widget(self._btn_export)
        right.addWidget(export_card)

        middle.addLayout(right, 2)
        root.addLayout(middle, 1)

        # ---- progress -----------------------------------------------------
        prog_card = SectionCard(c, "渲染进度")
        self._progress = AnimatedProgressBar(c)
        prog_card.add_widget(self._progress)
        self._prog_label = QLabel("就绪")
        self._prog_label.setProperty("cssClass", "mono")
        prog_card.add_widget(self._prog_label)
        root.addWidget(prog_card)

        # ---- engine wiring ------------------------------------------------
        self.request_prepare.connect(engine.prepare)
        self.request_render_anim.connect(engine.render_anim)
        self.request_export_anim.connect(engine.export_anim)

        engine.prepared.connect(self._on_prepared)
        engine.anim_frame.connect(self._on_anim_frame)
        engine.anim_progress.connect(self._on_anim_progress)
        engine.anim_done.connect(self._on_anim_done)
        engine.anim_exported.connect(self._on_exported)
        engine.error.connect(self._on_error)

    # ------------------------------------------------------------------
    def _on_input(self, path: str) -> None:
        self._prepared = False
        self._frames = []
        self._btn_export.setEnabled(False)
        self._awaiting_prepare = True
        self._prog_label.setText("正在重建 3D 场景…")
        self.request_prepare.emit(path, 0)

    def _on_prepared(self, info: dict) -> None:
        if not self._awaiting_prepare:
            return
        self._awaiting_prepare = False
        self._prepared = True
        self._prog_label.setText("场景就绪 · 点击「生成动画」")
        self.status_message.emit(f"场景重建完成 · {info['n_gaussians']:,} 高斯")

    def _on_render(self) -> None:
        if not self._prepared or self._rendering:
            return
        self._rendering = True
        self._frames = []
        self._btn_export.setEnabled(False)
        self._btn_render.setEnabled(False)
        self._progress.set_value(0.0)
        opts = {
            "type": self._picker.value(),
            "max_disparity": self._s_disparity.value(),
            "max_zoom": self._s_zoom.value(),
            "num_steps": self._steps.value(),
            "num_repeats": self._repeats.value(),
            "preview_width": ANIM_PREVIEW_WIDTH,
        }
        self.request_render_anim.emit(opts)

    def _on_anim_frame(self, frame: object) -> None:
        self._frames.append(frame)

    def _on_anim_progress(self, i: int, total: int) -> None:
        self._progress.set_value(i / total if total else 0.0)
        self._prog_label.setText(f"渲染中 {i}/{total}")

    def _on_anim_done(self, result: dict) -> None:
        self._rendering = False
        self._btn_render.setEnabled(True)
        self._progress.set_value(1.0)
        n = result["n_frames"]
        self._frame_info.setText(f"{n} 帧")
        self._prog_label.setText(f"完成 · {n} 帧")
        if n > 0:
            self._btn_export.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._rendering = False
        self._awaiting_prepare = False
        self._btn_render.setEnabled(True)
        if self._frames:
            self._btn_export.setEnabled(True)
        self._prog_label.setText("出错")
        self.status_message.emit(msg)

    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        if not self._frames:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "导出动画", str(Path.home() / "parallax.mp4"),
            "视频 (*.mp4)",
        )
        if not path:
            return
        codec_map = {"H.264": "h264", "H.265": "h265", "AV1": "av1"}
        codec = codec_map[self._codec.currentText()]
        fps = int(self._fps.currentText())
        self.status_message.emit(f"正在导出 {path} …")
        self._prog_label.setText("正在导出视频…")
        self._btn_export.setEnabled(False)
        self.request_export_anim.emit({
            "path": path, "codec": codec, "fps": fps, "frames": self._frames,
        })

    def _on_exported(self, path: str) -> None:
        self._btn_export.setEnabled(True)
        self._prog_label.setText(f"已导出 → {path}"[:80])
        self.status_message.emit(f"动画已导出 → {path}")

    # ------------------------------------------------------------------
    def apply_theme(self, c: Colors) -> None:
        self._progress._colors = c
        self._picker.set_colors(c)
        for s in (self._s_disparity, self._s_zoom):
            s.set_colors(c)
