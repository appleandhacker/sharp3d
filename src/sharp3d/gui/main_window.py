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

        # Adaptive initial size: fits any display (2K/4K) without exceeding
        # screen bounds. Hard-capped to available geometry so it NEVER overflows.
        from PySide6.QtWidgets import QApplication
        screen_geo = QApplication.primaryScreen().availableGeometry()
        # Hard limit: window can never be larger than the screen
        self.setMaximumSize(screen_geo.width(), screen_geo.height())
        # Initial size: 70% width, 75% height, clamped to [860x580, 1800x1100]
        w = max(860, min(int(screen_geo.width() * 0.70), 1800))
        h = max(580, min(int(screen_geo.height() * 0.75), 1100))
        self.resize(w, h)
        # Center on screen
        self.move(
            screen_geo.x() + (screen_geo.width() - w) // 2,
            screen_geo.y() + (screen_geo.height() - h) // 2,
        )

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
