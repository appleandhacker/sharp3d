"""Gaussian viewer tab — load a .ply and orbit around the 3D scene."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .theme import Colors, ThemeManager
from .widgets import FileField, PreviewPane, SectionCard, StereoSlider
from .worker import EngineProcess

PLY_FILTER = "Gaussian PLY (*.ply);;所有文件 (*)"


class GaussianTab(QWidget):
    """Interactive 3D Gaussian splat viewer with orbit controls."""

    status_message = Signal(str)
    request_load_ply = Signal(str)
    request_render_orbit = Signal(dict)

    def __init__(self, theme: ThemeManager, engine: EngineProcess, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._engine = engine
        self._loaded = False
        self._auto_rotate = False
        self._azimuth = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(12)
        c = theme.colors

        # ---- IO card ------------------------------------------------------
        io_card = SectionCard(c, "高斯文件")
        self._input = FileField(c, "PLY", file_filter=PLY_FILTER)
        self._input.path_selected.connect(self._on_load)
        io_card.add_widget(self._input)
        self._info_label = QLabel("未加载")
        self._info_label.setProperty("cssClass", "hint")
        io_card.add_widget(self._info_label)
        root.addWidget(io_card)

        # ---- middle: preview + controls -----------------------------------
        middle = QHBoxLayout()
        middle.setSpacing(12)

        preview_card = SectionCard(c, "3D 预览")
        self._preview = PreviewPane(c, stereo=False)
        self._preview.set_message("加载 PLY 文件\n拖入或点击浏览")
        self._preview.file_dropped.connect(self._input.set_path)
        preview_card.add_widget(self._preview)

        ctrl = QHBoxLayout()
        self._btn_render = QPushButton("渲染当前视角")
        self._btn_render.setEnabled(False)
        self._btn_render.clicked.connect(self._render)
        self._chk_auto = QCheckBox("自动旋转")
        self._chk_auto.toggled.connect(self._on_auto_toggled)
        self._btn_export = QPushButton("导出 PNG…")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        ctrl.addWidget(self._btn_render)
        ctrl.addWidget(self._chk_auto)
        ctrl.addStretch(1)
        ctrl.addWidget(self._btn_export)
        preview_card.add_layout(ctrl)
        middle.addWidget(preview_card, 3)

        # right column — orbit controls
        right = QVBoxLayout()
        right.setSpacing(12)

        orbit_card = SectionCard(c, "视角控制")
        self._s_azimuth = StereoSlider(c, "方位角", -180.0, 180.0, 0.0,
                                       fmt="{:.0f}", unit="°")
        self._s_azimuth.value_changed.connect(self._on_slider)
        self._s_elevation = StereoSlider(c, "仰角", -60.0, 60.0, 0.0,
                                         fmt="{:.0f}", unit="°")
        self._s_elevation.value_changed.connect(self._on_slider)
        self._s_distance = StereoSlider(c, "距离", 1.0, 20.0, 5.0,
                                        fmt="{:.1f}", unit="")
        self._s_distance.value_changed.connect(self._on_slider)
        orbit_card.add_widget(self._s_azimuth)
        orbit_card.add_widget(self._s_elevation)
        orbit_card.add_widget(self._s_distance)
        right.addWidget(orbit_card)

        middle.addLayout(right, 2)
        root.addLayout(middle, 1)

        # ---- auto-rotate timer -------------------------------------------
        self._rotate_timer = QTimer(self)
        self._rotate_timer.timeout.connect(self._auto_step)

        # ---- engine wiring ------------------------------------------------
        self.request_load_ply.connect(engine.load_ply)
        self.request_render_orbit.connect(engine.render_orbit)
        engine.ply_loaded.connect(self._on_ply_loaded)
        engine.orbit_frame.connect(self._on_orbit_frame)
        engine.error.connect(self._on_error)

    # ------------------------------------------------------------------
    def _on_load(self, path: str) -> None:
        self._loaded = False
        self._preview.clear_image()
        self._preview.set_message("正在加载 PLY…")
        self._info_label.setText("加载中…")
        self.request_load_ply.emit(path)

    def _on_ply_loaded(self, info: dict) -> None:
        self._loaded = True
        self._btn_render.setEnabled(True)
        n = info.get("n_gaussians", 0)
        w = info.get("width", 0)
        h = info.get("height", 0)
        self._info_label.setText(f"{n:,} 高斯点 · 原始 {w}×{h}")
        self._preview.set_message("已加载 · 点击「渲染当前视角」")
        self.status_message.emit(f"PLY 已加载 · {n:,} 高斯点")
        # Auto-render the default view
        self._render()

    def _on_slider(self, _v: float) -> None:
        if self._loaded and not self._auto_rotate:
            self._render()

    def _render(self) -> None:
        if not self._loaded:
            return
        self._azimuth = self._s_azimuth.value()
        self.request_render_orbit.emit({
            "azimuth": self._s_azimuth.value(),
            "elevation": self._s_elevation.value(),
            "distance": self._s_distance.value(),
            "render_width": 960,
        })

    def _on_orbit_frame(self, frame) -> None:
        self._preview.set_image(frame)
        self._btn_export.setEnabled(True)
        self._last_frame = frame

    def _on_auto_toggled(self, checked: bool) -> None:
        self._auto_rotate = checked
        if checked:
            self._rotate_timer.start(80)  # ~12 fps rotation
        else:
            self._rotate_timer.stop()

    def _auto_step(self) -> None:
        self._azimuth = (self._azimuth + 3.0) % 360.0
        # Map 0-360 to slider range -180..180
        slider_val = self._azimuth if self._azimuth <= 180 else self._azimuth - 360
        self._s_azimuth.set_value(slider_val)
        self._render()

    def _on_export(self) -> None:
        if not hasattr(self, '_last_frame') or self._last_frame is None:
            return
        from PySide6.QtWidgets import QFileDialog
        from PIL import Image

        path, _ = QFileDialog.getSaveFileName(
            self, "导出截图", str(Path.home() / "gaussian_view.png"),
            "PNG (*.png)",
        )
        if not path:
            return
        Image.fromarray(self._last_frame).save(path)
        self.status_message.emit(f"已导出 → {path}")

    def _on_error(self, msg: str) -> None:
        self._preview.set_message("加载/渲染出错")
        self.status_message.emit(msg)

    # ------------------------------------------------------------------
    def apply_theme(self, c: Colors) -> None:
        self._preview.set_colors(c)
        for s in (self._s_azimuth, self._s_elevation, self._s_distance):
            s.set_colors(c)
