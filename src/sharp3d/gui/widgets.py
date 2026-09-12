"""Custom widgets for sharp3d GUI.

Hand-painted, animated controls that give the app its living feel:
  - AnimatedProgressBar: red→cyan gradient fill with a moving shimmer
  - GpuMeter: live GPU utilization + VRAM readout (pynvml)
  - StereoSlider: label + slider + monospace value readout
  - PreviewPane: SBS preview surface with red/cyan eye tags + empty state
  - SectionCard: titled panel container
  - FileField: read-only path + browse button, drag & drop aware
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    Property,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .theme import DISPLAY_FONT, MONO_FONT, Colors
from .i18n import tr


# ---------------------------------------------------------------------------
# Animated progress bar
# ---------------------------------------------------------------------------
class AnimatedProgressBar(QWidget):
    """Progress bar with a red→cyan gradient fill and a moving shimmer."""

    def __init__(self, colors: Colors, parent=None) -> None:
        super().__init__(parent)
        self._colors = colors
        self._value = 0.0          # 0..1
        self._busy = False         # indeterminate shimmer
        self._shimmer = 0.0        # 0..1 position of the highlight sweep
        self.setMinimumHeight(14)
        self.setMaximumHeight(14)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30fps

    def _tick(self) -> None:
        self._shimmer = (self._shimmer + 0.02) % 1.0
        if self._busy or self._value > 0:
            self.update()

    def set_value(self, frac: float) -> None:
        self._value = max(0.0, min(1.0, frac))
        self.update()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.update()

    def reset(self) -> None:
        self._value = 0.0
        self._busy = False
        self.update()

    def paintEvent(self, event) -> None:
        c = self._colors
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        track = QRectF(0, 0, w, h)

        # track
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c.track))
        p.drawRoundedRect(track, h / 2, h / 2)

        # fill
        fill_w = w * self._value
        if self._busy:
            fill_w = w  # full bar, shimmer sweeps
        if fill_w > 1:
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, QColor(c.red))
            grad.setColorAt(1.0, QColor(c.cyan))
            p.setBrush(grad)
            fill_rect = QRectF(0, 0, fill_w, h)
            p.drawRoundedRect(fill_rect, h / 2, h / 2)

        # shimmer highlight
        if self._busy or (0 < self._value < 1):
            sx = self._shimmer * (w + 120) - 60
            sg = QLinearGradient(sx - 60, 0, sx + 60, 0)
            hi = QColor("#ffffff")
            hi.setAlphaF(0.0)
            hi_mid = QColor("#ffffff")
            hi_mid.setAlphaF(0.18)
            sg.setColorAt(0.0, hi)
            sg.setColorAt(0.5, hi_mid)
            sg.setColorAt(1.0, hi)
            p.setBrush(sg)
            p.setClipPath(self._rounded_path(track, h / 2))
            p.drawRect(QRectF(sx - 60, 0, 120, h))
            p.setClipping(False)

        p.end()

    @staticmethod
    def _rounded_path(rect: QRectF, radius: float) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        return path


# ---------------------------------------------------------------------------
# GPU meter
# ---------------------------------------------------------------------------
class GpuMeter(QWidget):
    """Live GPU utilization + VRAM readout via pynvml."""

    def __init__(self, colors: Colors, parent=None) -> None:
        super().__init__(parent)
        self._colors = colors

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._name = QLabel("GPU —")
        self._name.setProperty("cssClass", "mono")
        self._util = QLabel("—%")
        self._util.setProperty("cssClass", "mono")
        self._vram = QLabel("VRAM —")
        self._vram.setProperty("cssClass", "mono")

        self._bar = _MiniBar(colors)

        layout.addWidget(self._name, 1)
        layout.addWidget(self._bar, 0)
        layout.addWidget(self._util, 0)
        layout.addWidget(self._vram, 0)

        self._nvml_ok = False
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "ignore")
            self._name.setText(name)
        except Exception:
            self._name.setText(tr("GPU 不可用"))

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(1000)
        self._poll()

    def _poll(self) -> None:
        if not self._nvml_ok:
            return
        try:
            import pynvml

            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._util.setText(f"{util.gpu}%")
            self._bar.set_value(util.gpu / 100.0)
            used_gb = mem.used / 1024**3
            total_gb = mem.total / 1024**3
            self._vram.setText(f"VRAM {used_gb:.1f}/{total_gb:.0f}GB")
        except Exception:
            pass

    def set_colors(self, colors: Colors) -> None:
        self._colors = colors
        self._bar.set_colors(colors)


class _MiniBar(QWidget):
    """Tiny horizontal utilization bar."""

    def __init__(self, colors: Colors, parent=None) -> None:
        super().__init__(parent)
        self._colors = colors
        self._value = 0.0
        self.setFixedSize(90, 8)

    def set_value(self, v: float) -> None:
        self._value = max(0.0, min(1.0, v))
        self.update()

    def set_colors(self, colors: Colors) -> None:
        self._colors = colors
        self.update()

    def paintEvent(self, event) -> None:
        c = self._colors
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c.track))
        p.drawRoundedRect(0, 0, self.width(), self.height(), 4, 4)
        if self._value > 0.01:
            grad = QLinearGradient(0, 0, self.width(), 0)
            grad.setColorAt(0.0, QColor(c.cyan))
            grad.setColorAt(1.0, QColor(c.red))
            p.setBrush(grad)
            p.drawRoundedRect(0, 0, self.width() * self._value, self.height(), 4, 4)
        p.end()


# ---------------------------------------------------------------------------
# Stereo slider (label + slider + value readout)
# ---------------------------------------------------------------------------
class StereoSlider(QWidget):
    """A labeled slider with a live monospace value readout."""

    value_changed = Signal(float)

    def __init__(
        self,
        colors: Colors,
        label: str,
        lo: float,
        hi: float,
        default: float,
        fmt: str = "{:.2f}",
        unit: str = "",
        stereo: bool = False,
        red_handle: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._colors = colors
        self._lo, self._hi = lo, hi
        self._fmt, self._unit = fmt, unit

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self._label = QLabel(label)
        self._label.setStyleSheet(f"font-weight:600; font-size:12px; color:{colors.text};")
        self._readout = QLabel()
        self._readout.setFont(QFont(MONO_FONT, 10))
        self._readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self._label)
        top.addStretch(1)
        top.addWidget(self._readout)
        layout.addLayout(top)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        if stereo:
            self._slider.setProperty("cssClass", "stereo")
        if red_handle:
            self._slider.setProperty("red", "true")
        self._slider.valueChanged.connect(self._on_slide)
        layout.addWidget(self._slider)

        self.set_value(default)

    def _on_slide(self, _v: int) -> None:
        val = self.value()
        self._readout.setText((self._fmt + self._unit).format(val))
        self.value_changed.emit(val)

    def value(self) -> float:
        frac = self._slider.value() / 1000.0
        return self._lo + frac * (self._hi - self._lo)

    def set_value(self, val: float) -> None:
        frac = (val - self._lo) / (self._hi - self._lo)
        self._slider.setValue(int(round(frac * 1000)))
        self._readout.setText((self._fmt + self._unit).format(val))

    def set_colors(self, colors: Colors) -> None:
        self._colors = colors
        self._label.setStyleSheet(f"font-weight:600; font-size:12px; color:{colors.text};")


# ---------------------------------------------------------------------------
# Section card
# ---------------------------------------------------------------------------
class SectionCard(QFrame):
    """A titled panel. Children are added to its inner layout."""

    def __init__(self, colors: Colors, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)  # picks up card QSS
        self._colors = colors

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 13, 16, 15)
        outer.setSpacing(10)

        self._title = QLabel(title)
        self._title.setProperty("cssClass", "cardTitle")
        outer.addWidget(self._title)

        self.body = QVBoxLayout()
        self.body.setSpacing(9)
        outer.addLayout(self.body)

    def add_widget(self, w: QWidget) -> None:
        self.body.addWidget(w)

    def add_layout(self, lay) -> None:
        self.body.addLayout(lay)


# ---------------------------------------------------------------------------
# File field (read-only path + browse, drag & drop aware)
# ---------------------------------------------------------------------------
class FileField(QWidget):
    """Read-only path display + browse button. Emits path_selected."""

    path_selected = Signal(str)

    def __init__(
        self,
        colors: Colors,
        label: str,
        pick_dir: bool = False,
        file_filter: str = "",
        save: bool = False,
        allow_folder: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._colors = colors
        self._pick_dir = pick_dir
        self._filter = file_filter
        self._save = save
        self._allow_folder = allow_folder
        self.setAcceptDrops(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._label = QLabel(label)
        self._label.setFixedWidth(58)
        self._label.setStyleSheet(f"font-weight:600; font-size:12px; color:{colors.text_muted};")

        self._edit = QLineEdit()
        self._edit.setReadOnly(True)
        self._edit.setPlaceholderText(tr("拖入文件/文件夹，或点击浏览…"))

        self._btn = QPushButton(tr("浏览…"))
        self._btn.setFixedWidth(64)
        self._btn.clicked.connect(self._browse)

        layout.addWidget(self._label)
        layout.addWidget(self._edit, 1)
        layout.addWidget(self._btn)

    def _browse(self) -> None:
        start = self._edit.text() or str(Path.home())
        if self._pick_dir:
            path = QFileDialog.getExistingDirectory(self, tr("选择目录"), start)
        elif self._save:
            path, _ = QFileDialog.getSaveFileName(self, tr("保存"), start, self._filter)
        elif self._allow_folder:
            # Offer both file and folder selection via menu
            from PySide6.QtWidgets import QMenu
            from PySide6.QtCore import QPoint
            menu = QMenu(self)
            act_file = menu.addAction(tr("选择文件"))
            act_folder = menu.addAction(tr("选择文件夹（批量）"))
            chosen = menu.exec(self._btn.mapToGlobal(
                QPoint(0, self._btn.height())))
            if chosen == act_file:
                path, _ = QFileDialog.getOpenFileName(self, tr("打开"), start, self._filter)
            elif chosen == act_folder:
                path = QFileDialog.getExistingDirectory(self, tr("选择文件夹（批量）"), start)
            else:
                return
        else:
            path, _ = QFileDialog.getOpenFileName(self, tr("打开"), start, self._filter)
        if path:
            self.set_path(path)

    def set_path(self, path: str) -> None:
        self._edit.setText(path)
        self.path_selected.emit(path)

    def path(self) -> str:
        return self._edit.text().strip()

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:
        urls = e.mimeData().urls()
        if urls:
            self.set_path(urls[0].toLocalFile())


# ---------------------------------------------------------------------------
# Preview pane (SBS surface with eye tags + empty state)
# ---------------------------------------------------------------------------
class PreviewPane(QWidget):
    """Displays an SBS image (left|right) with red/cyan eye tags.

    Shows a distinctive empty state when nothing is loaded.
    """

    def __init__(self, colors: Colors, stereo: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._colors = colors
        self._stereo = stereo
        self._pixmap: QPixmap | None = None
        self._message = tr("拖入图片或视频\n开始立体转换")
        self.setMinimumSize(480, 260)
        self.setAcceptDrops(True)

    def set_stereo(self, stereo: bool) -> None:
        self._stereo = stereo
        self.update()

    def set_colors(self, colors: Colors) -> None:
        self._colors = colors
        self.update()

    def set_image(self, rgb: np.ndarray) -> None:
        """rgb: (H, W, 3) uint8 numpy array (full SBS, left|right)."""
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage

        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def clear_image(self) -> None:
        self._pixmap = None
        self.update()

    def set_message(self, msg: str) -> None:
        self._message = msg
        self.update()

    def paintEvent(self, event) -> None:
        c = self._colors
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        w, h = self.width(), self.height()

        # background
        p.fillRect(0, 0, w, h, QColor(c.bg_alt))
        p.setPen(QPen(QColor(c.border), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(0.5, 0.5, w - 1, h - 1, 10, 10)

        if self._pixmap is None:
            self._draw_empty(p, w, h)
            p.end()
            return

        # fit pixmap into pane with margin
        margin = 14
        avail_w, avail_h = w - margin * 2, h - margin * 2 - 24  # leave room for tags
        pm = self._pixmap
        scaled = pm.scaled(avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (w - scaled.width()) / 2
        y = (h - scaled.height()) / 2 + 8
        target = QRectF(x, y, scaled.width(), scaled.height())
        p.drawPixmap(target.toRect(), scaled)

        # center divider + eye tags only in stereo mode
        if self._stereo:
            cx = x + scaled.width() / 2
            pen = QPen(QColor(c.border_strong), 1, Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(cx, y + 4), QPointF(cx, y + scaled.height() - 4))
            self._draw_tag(p, x + 8, y + 8, "L", c.red)
            self._draw_tag(p, cx + 8, y + 8, "R", c.cyan)

        p.end()

    def _draw_empty(self, p: QPainter, w: int, h: int) -> None:
        c = self._colors
        cx, cy = w / 2, h / 2

        # two overlapping rounded rects suggesting stereo (red + cyan offset)
        bw, bh = 74, 48
        off = 10
        p.setPen(QPen(QColor(c.red), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(cx - bw - off / 2, cy - bh - 34, bw, bh), 8, 8)
        p.setPen(QPen(QColor(c.cyan), 2))
        p.drawRoundedRect(QRectF(cx + off / 2, cy - bh - 34, bw, bh), 8, 8)

        p.setPen(QColor(c.text_faint))
        f = QFont(DISPLAY_FONT, 13, QFont.Weight.Medium)
        p.setFont(f)
        lines = self._message.split("\n")
        ty = cy + 34
        for line in lines:
            p.drawText(QRectF(0, ty, w, 22), Qt.AlignHCenter, line)
            ty += 22

    def _draw_tag(self, p: QPainter, x: float, y: float, letter: str, color: str) -> None:
        c = self._colors
        p.setPen(Qt.NoPen)
        tag_bg = QColor(color)
        tag_bg.setAlphaF(0.85)
        p.setBrush(tag_bg)
        p.drawRoundedRect(QRectF(x, y, 22, 18), 5, 5)
        p.setPen(QColor("#0e1013"))
        f = QFont(DISPLAY_FONT, 10, QFont.Weight.Bold)
        p.setFont(f)
        p.drawText(QRectF(x, y, 22, 18), Qt.AlignCenter, letter)

    # drag & drop passthrough to parent tab
    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:
        urls = e.mimeData().urls()
        if urls:
            # let the tab handle it via a signal
            self.file_dropped.emit(urls[0].toLocalFile())

    file_dropped = Signal(str)
