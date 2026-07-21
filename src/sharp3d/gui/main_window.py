"""Main window — tabs, status bar, theme management, engine lifecycle."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .anim_tab import AnimTab
from .sbs_tab import SbsTab
from .theme import DISPLAY_FONT, ThemeManager, build_palette, build_qss
from .widgets import GpuMeter
from .worker import EngineProcess


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("sharp3d — 2D → 3D 立体转换")

        # Adaptive initial size: proportional to logical screen (DPI-aware).
        # Qt6 returns logical pixels from availableGeometry(), so this works
        # correctly at any Windows scaling (100%/150%/200%).
        # 4K@200%→logical 1920x1080→1152x756, 4K@100%→3840x2160→1600x1000,
        # 2K@150%→1707x960→1024x672, 1080p→1920x1080→1152x756.
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QRect
        geo = QApplication.primaryScreen().availableGeometry()
        w = max(900, min(int(geo.width() * 0.60), 1600))
        h = max(600, min(int(geo.height() * 0.70), 1000))
        x = geo.x() + (geo.width() - w) // 2
        y = geo.y() + (geo.height() - h) // 2
        self.setGeometry(QRect(x, y, w, h))

        self._theme = ThemeManager()
        self._engine = EngineProcess()

        # central
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # header band with app identity
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 14, 20, 10)
        header_layout.setSpacing(12)

        # stereo logotype: offset red/cyan "3D"
        logo = QLabel("◐◑")
        logo.setFont(QFont(DISPLAY_FONT, 20, QFont.Weight.Bold))
        logo.setStyleSheet(
            f"color:{self._theme.colors.red};"
        )
        title = QLabel("sharp3d")
        title.setFont(QFont(DISPLAY_FONT, 22, QFont.Weight.Bold))
        subtitle = QLabel("平面照片 / 视频 → 立体 3D")
        subtitle.setProperty("cssClass", "hint")
        subtitle.setStyleSheet("font-size:12px;")

        title_block = QVBoxLayout()
        title_block.setSpacing(0)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        header_layout.addWidget(logo)
        header_layout.addLayout(title_block)
        header_layout.addStretch(1)

        self._gpu = GpuMeter(self._theme.colors)
        header_layout.addWidget(self._gpu)

        root.addWidget(header)

        # tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._sbs = SbsTab(self._theme, self._engine)
        self._anim = AnimTab(self._theme, self._engine)
        self._tabs.addTab(self._sbs, "SBS 立体转换")
        self._tabs.addTab(self._anim, "2.5D 视差动画")
        root.addWidget(self._tabs, 1)

        # status bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_label = QLabel("就绪")
        sb.addWidget(self._status_label, 1)
        self._theme_label = QLabel("")
        sb.addPermanentWidget(self._theme_label)

        # signals
        self._theme.changed.connect(self._apply_theme)
        self._engine.status.connect(self._status_label.setText)
        self._engine.error.connect(self._status_label.setText)
        self._sbs.status_message.connect(self._status_label.setText)
        self._anim.status_message.connect(self._status_label.setText)

        self._apply_theme(self._theme.is_dark)

        # Start loading the model immediately: the one-time load/compile cost
        # (~1 min on first run) overlaps with the user picking a file, instead
        # of stalling the first conversion.
        self._engine.preload()

    # ------------------------------------------------------------------
    def _apply_theme(self, is_dark: bool) -> None:
        c = self._theme.colors
        app = self.window().parent()
        from PySide6.QtWidgets import QApplication

        qapp = QApplication.instance()
        qapp.setPalette(build_palette(c))
        qapp.setStyleSheet(build_qss(c))
        self._gpu.set_colors(c)
        self._sbs.apply_theme(c)
        self._anim.apply_theme(c)
        self._theme_label.setText("暗色模式" if is_dark else "亮色模式")

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._engine.stop()
        super().closeEvent(event)
