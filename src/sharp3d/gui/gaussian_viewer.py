"""Standalone Gaussian splat viewer window with mouse-drag orbit controls.

Opens as a separate window (not a tab). Drag to rotate, scroll to zoom.
Renders via the engine's render_orbit method (gsplat on GPU).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QImage, QPixmap, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QToolBar,
    QWidget,
)

from .theme import Colors
from .worker import EngineProcess
from .i18n import tr

PLY_FILTER = tr("Gaussian PLY (*.ply);;所有文件 (*)")


class OrbitView(QWidget):
    """Interactive orbit viewport: drag to rotate, scroll to zoom, drop PLY to load."""

    view_changed = Signal(float, float, float)  # d_azimuth, d_elevation, d_distance
    file_dropped = Signal(str)  # path to dropped .ply file
    drag_started = Signal()
    drag_ended = Signal()

    def __init__(self, colors: Colors, parent=None) -> None:
        super().__init__(parent)
        self._colors = colors
        self._pixmap: QPixmap | None = None
        self._message = tr("拖入 PLY 文件打开\n左键旋转 · 滚轮缩放")
        self._dragging = False
        self._last_pos = QPoint()
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setAcceptDrops(True)

    # ---- display --------------------------------------------------------
    def set_image(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def clear_image(self) -> None:
        self._pixmap = None
        self.update()

    def set_message(self, msg: str) -> None:
        self._message = msg
        self._pixmap = None
        self.update()

    def set_colors(self, colors: Colors) -> None:
        self._colors = colors
        self.update()

    # ---- mouse interaction ----------------------------------------------
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self.drag_started.emit()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if not self._dragging:
            return
        delta = e.pos() - self._last_pos
        self._last_pos = e.pos()
        self.view_changed.emit(delta.x() * 0.3, -delta.y() * 0.3, 0.0)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.OpenHandCursor)
            self.drag_ended.emit()

    def wheelEvent(self, e: QWheelEvent) -> None:
        delta = -e.angleDelta().y() / 120.0 * 0.5
        self.view_changed.emit(0.0, 0.0, delta)

    # ---- drag & drop ----------------------------------------------------
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                if url.toLocalFile().lower().endswith(".ply"):
                    e.acceptProposedAction()
                    return

    def dropEvent(self, e) -> None:
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".ply"):
                self.file_dropped.emit(path)
                return

    # ---- painting -------------------------------------------------------
    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QPainter, QColor, QFont
        from .theme import DISPLAY_FONT

        c = self._colors
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, QColor(c.bg_alt))

        if self._pixmap is None:
            p.setPen(QColor(c.text_faint))
            p.setFont(QFont(DISPLAY_FONT, 14))
            lines = self._message.split("\n")
            cy = h // 2 - len(lines) * 12
            for line in lines:
                p.drawText(0, cy, w, 28, Qt.AlignHCenter, line)
                cy += 28
            p.end()
            return

        margin = 8
        scaled = self._pixmap.scaled(
            w - margin * 2, h - margin * 2,
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (w - scaled.width()) // 2
        y = (h - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        p.end()


class GaussianViewerWindow(QMainWindow):
    """Standalone Gaussian splat viewer with orbit controls."""

    def __init__(self, engine: EngineProcess, colors: Colors, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("高斯查看器 — sharp3d"))
        self.resize(900, 680)
        self._engine = engine
        self._colors = colors
        self._loaded = False

        # Orbit state (azimuth=180 → camera at -Z looking toward +Z = scene front)
        self._azimuth = 180.0
        self._elevation = 0.0
        self._distance = 5.0
        self._render_pending = False

        # Central viewport
        self._view = OrbitView(colors)
        self._view.view_changed.connect(self._on_view_changed)
        self._view.file_dropped.connect(self._on_file_dropped)
        self._view.drag_started.connect(self._on_drag_started)
        self._view.drag_ended.connect(self._on_drag_ended)
        self.setCentralWidget(self._view)

        # Toolbar
        tb = QToolBar(tr("工具"))
        tb.setMovable(False)
        self.addToolBar(tb)
        btn_open = QPushButton(tr("打开 PLY…"))
        btn_open.clicked.connect(self._on_open)
        tb.addWidget(btn_open)
        btn_reset = QPushButton(tr("重置视角"))
        btn_reset.clicked.connect(self._on_reset)
        tb.addWidget(btn_reset)
        self._info = QLabel(tr("  未加载"))
        tb.addWidget(self._info)

        # Render throttle: continuous during drag, single-shot for wheel
        from PySide6.QtCore import QTimer
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._do_render)
        self._render_timer.setInterval(33)
        self._dragging = False

        # Engine signals
        engine.ply_loaded.connect(self._on_ply_loaded)
        engine.orbit_frame.connect(self._on_orbit_frame)

    # ---- actions --------------------------------------------------------
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("打开高斯 PLY 文件"), str(Path.home()), PLY_FILTER)
        if path:
            self._view.set_message(tr("加载中…"))
            self._engine.load_ply(path)

    def _on_file_dropped(self, path: str) -> None:
        self._view.set_message(tr("加载中…"))
        self._engine.load_ply(path)

    def _on_reset(self) -> None:
        self._azimuth = 180.0
        self._elevation = 0.0
        self._distance = 5.0
        self._do_render()

    def _on_view_changed(self, d_azimuth: float, d_elevation: float,
                         d_distance: float) -> None:
        self._azimuth += d_azimuth
        self._elevation = max(-80.0, min(80.0, self._elevation + d_elevation))
        self._distance = max(1.0, min(30.0, self._distance + d_distance))
        # For wheel events (not dragging): single-shot render
        if not self._dragging and not self._render_timer.isActive():
            self._render_timer.start()

    def _on_drag_started(self) -> None:
        self._dragging = True
        if self._loaded:
            self._render_timer.start()  # continuous 30fps during drag

    def _on_drag_ended(self) -> None:
        self._dragging = False
        # One final render to capture last position, then stop
        self._do_render()
        self._render_timer.stop()

    def _do_render(self) -> None:
        """Render current orbit state. Timer keeps running during drag."""
        if not self._loaded:
            self._render_timer.stop()
            return
        self._engine.render_orbit({
            "azimuth": self._azimuth,
            "elevation": self._elevation,
            "distance": self._distance,
            "render_width": 960,
        })
        if not self._dragging:
            self._render_timer.stop()

    # ---- engine callbacks -----------------------------------------------
    def _on_ply_loaded(self, info: dict) -> None:
        self._loaded = True
        n = info.get("n_gaussians", 0)
        self._info.setText(tr("  {:,} 高斯点 · 拖拽旋转 · 滚轮缩放").format(n))
        self._azimuth = 180.0
        self._elevation = 0.0
        self._distance = 5.0
        self._do_render()

    def _on_orbit_frame(self, frame) -> None:
        self._view.set_image(frame)

    # ---- theme ----------------------------------------------------------
    def set_colors(self, c: Colors) -> None:
        self._colors = c
        self._view.set_colors(c)

    # ---- lifecycle ------------------------------------------------------
    def closeEvent(self, event) -> None:
        """Disconnect engine signals to prevent crash on deleted Qt object."""
        self._render_timer.stop()
        try:
            self._engine.ply_loaded.disconnect(self._on_ply_loaded)
            self._engine.orbit_frame.disconnect(self._on_orbit_frame)
        except RuntimeError:
            pass  # already disconnected
        super().closeEvent(event)
