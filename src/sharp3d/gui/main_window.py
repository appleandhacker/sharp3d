"""Main window — tabs, status bar, theme management, engine lifecycle."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .anim_tab import AnimTab
from .gaussian_viewer import GaussianViewerWindow
from .sbs_tab import SbsTab, precision_text
from .vr_tab import VrTab
from .theme import DISPLAY_FONT, ThemeManager, build_palette, build_qss
from .widgets import GpuMeter
from .worker import EngineProcess
from .. import __version__
from .i18n import current_language, set_language, tr


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"sharp3d v{__version__}")

        # Window icon
        import sys as _sys
        from pathlib import Path
        if getattr(_sys, "frozen", False):
            icon_path = Path(_sys._MEIPASS) / "assets" / "sharp3d_icon.ico"
        else:
            icon_path = Path(__file__).resolve().parents[3] / "assets" / "sharp3d_icon.svg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Adaptive initial size: proportional to logical screen (DPI-aware).
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QRect
        geo = QApplication.primaryScreen().availableGeometry()
        w = max(800, min(int(geo.width() * 0.50), 1200))
        h = max(480, min(int(geo.height() * 0.60), 800))
        x = geo.x() + (geo.width() - w) // 2
        y = geo.y() + (geo.height() - h) // 2
        self.setGeometry(QRect(x, y, w, h))

        self._theme = ThemeManager()
        self._engine = EngineProcess()
        self._viewer: GaussianViewerWindow | None = None

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

        logo = QLabel("◐◑")
        logo.setFont(QFont(DISPLAY_FONT, 20, QFont.Weight.Bold))
        logo.setStyleSheet(f"color:{self._theme.colors.red};")
        title = QLabel("sharp3d")
        title.setFont(QFont(DISPLAY_FONT, 22, QFont.Weight.Bold))

        header_layout.addWidget(logo)
        header_layout.addWidget(title)
        header_layout.addStretch(1)

        self._gpu = GpuMeter(self._theme.colors)
        header_layout.addWidget(self._gpu)

        # language switcher (takes effect after restart)
        self._lang = QComboBox()
        self._lang.addItem("中文", "zh")
        self._lang.addItem("English", "en")
        self._lang.setCurrentIndex(1 if current_language() == "en" else 0)
        self._lang.currentIndexChanged.connect(self._on_language_changed)
        self._lang.setToolTip("切换界面语言 / Switch UI language (restart to apply)")
        header_layout.addWidget(self._lang)

        root.addWidget(header)

        # tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._sbs = SbsTab(self._theme, self._engine)
        self._anim = AnimTab(self._theme, self._engine)
        self._vr = VrTab(self._theme, self._engine)
        self._tabs.addTab(self._sbs, tr("平面立体转换"))
        self._tabs.addTab(self._vr, tr("全景转换"))
        self._tabs.addTab(self._anim, tr("2.5D 视差动画"))

        # Gaussian viewer button in tab bar corner
        btn_viewer = QPushButton(tr("高斯查看器"))
        btn_viewer.clicked.connect(self._open_viewer)
        self._tabs.setCornerWidget(btn_viewer)

        root.addWidget(self._tabs, 1)

        # status bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_label = QLabel(tr("正在加载模型…"))
        sb.addWidget(self._status_label, 1)
        self._precision_label = QLabel(tr("模型量化精度：未加载"))
        self._precision_label.setProperty("cssClass", "hint")
        sb.addPermanentWidget(self._precision_label)
        self._theme_label = QLabel("")
        sb.addPermanentWidget(self._theme_label)

        # signals
        self._theme.changed.connect(self._apply_theme)
        self._engine.status.connect(self._status_label.setText)
        # error now carries (message, job_id): the main window shows every
        # error regardless of owner, so drop the id.
        self._engine.error.connect(
            lambda msg, job_id=-1: self._status_label.setText(msg))
        self._engine.model_accel.connect(self._on_model_accel)
        self._sbs.status_message.connect(self._status_label.setText)
        self._anim.status_message.connect(self._status_label.setText)
        self._vr.status_message.connect(self._status_label.setText)

        self._apply_theme(self._theme.is_dark)

        # Start loading the model immediately
        self._engine.preload()

    # ------------------------------------------------------------------
    def _on_model_accel(self, status: list) -> None:
        self._precision_label.setText(tr("模型量化精度：{}").format(precision_text(status)))

    # ------------------------------------------------------------------
    def _open_viewer(self) -> None:
        """Open (or bring to front) the standalone Gaussian viewer window."""
        if self._viewer is None or not self._viewer.isVisible():
            self._viewer = GaussianViewerWindow(
                self._engine, self._theme.colors, parent=self)
        self._viewer.show()
        self._viewer.raise_()
        self._viewer.activateWindow()

    # ------------------------------------------------------------------
    def _on_language_changed(self, index: int) -> None:
        lang = self._lang.itemData(index)
        if lang == current_language():
            return
        set_language(lang)
        QMessageBox.information(
            self, tr("语言已切换"),
            tr("界面语言将在下次启动 sharp3d 时生效。"))

    def _apply_theme(self, is_dark: bool) -> None:
        c = self._theme.colors
        from PySide6.QtWidgets import QApplication

        qapp = QApplication.instance()
        qapp.setPalette(build_palette(c))
        qapp.setStyleSheet(build_qss(c))
        self._gpu.set_colors(c)
        self._sbs.apply_theme(c)
        self._anim.apply_theme(c)
        self._vr.apply_theme(c)
        if self._viewer is not None:
            self._viewer.set_colors(c)
        self._theme_label.setText(tr("暗色模式") if is_dark else tr("亮色模式"))

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self._viewer is not None:
            self._viewer.close()
        self._engine.stop()
        super().closeEvent(event)
