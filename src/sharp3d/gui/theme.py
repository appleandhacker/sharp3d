"""System theme detection + palettes + QSS for sharp3d GUI.

Visual identity: stereoscopic anaglyph theme.
  - Left eye  = red accent
  - Right eye = cyan accent
  The red/cyan pair is THE iconic color language of stereo 3D, and it also
  labels the two preview panes meaningfully.

Theme follows the Windows light/dark setting (registry AppsUseLightTheme),
polled every second so switching is live without restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPalette

# --- Typography -------------------------------------------------------------
# Bahnschrift: Windows' DIN-style face — technical, distinctive, great for
# display numerals and headers. Segoe UI for readable body text.
DISPLAY_FONT = "Bahnschrift"
BODY_FONT = "Segoe UI"
MONO_FONT = "Cascadia Mono"


@dataclass
class Colors:
    """Full color token set for one theme."""

    bg: str            # window background
    bg_alt: str        # subtle alternate background (tab strip, status bar)
    card: str          # card / panel surface
    raised: str        # raised control surface (buttons, inputs)
    hover: str         # hover surface
    border: str        # hairline border
    border_strong: str # emphasized border
    text: str          # primary text
    text_muted: str    # secondary text
    text_faint: str    # tertiary / placeholder
    red: str           # left-eye accent
    red_bright: str    # left-eye accent (hover / glow)
    cyan: str          # right-eye accent
    cyan_bright: str   # right-eye accent (hover / glow)
    track: str         # slider groove / progress track
    ok: str
    warn: str
    danger: str
    shadow: str        # drop shadow color


DARK = Colors(
    bg="#141619",
    bg_alt="#101214",
    card="#1b1e23",
    raised="#242830",
    hover="#2c313b",
    border="#2e333d",
    border_strong="#3d4451",
    text="#e9ecf1",
    text_muted="#a2a9b5",
    text_faint="#6b7280",
    red="#fb6f76",
    red_bright="#ff8d92",
    cyan="#33d6e2",
    cyan_bright="#5ce6f0",
    track="#2a2e36",
    ok="#3ecf8e",
    warn="#f5b455",
    danger="#f0645f",
    shadow="#000000",
)

LIGHT = Colors(
    bg="#eef0f4",
    bg_alt="#e6e9ee",
    card="#ffffff",
    raised="#f2f4f7",
    hover="#e8ebf0",
    border="#d7dbe2",
    border_strong="#c2c8d2",
    text="#1c2027",
    text_muted="#5b6270",
    text_faint="#9aa1ad",
    red="#d92d43",
    red_bright="#ef4458",
    cyan="#0891b2",
    cyan_bright="#0aa8cf",
    track="#dde1e8",
    ok="#12a06b",
    warn="#c07f1f",
    danger="#d43d38",
    shadow="#1c2027",
)


def is_dark_mode() -> bool:
    """Read Windows light/dark app-mode setting from the registry."""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        return val == 0
    except Exception:
        return True  # default to dark


class ThemeManager(QObject):
    """Watches the OS theme and emits `changed` when it flips."""

    changed = Signal(bool)  # True = now dark

    def __init__(self) -> None:
        super().__init__()
        self._dark = is_dark_mode()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(1000)

    def _poll(self) -> None:
        dark = is_dark_mode()
        if dark != self._dark:
            self._dark = dark
            self.changed.emit(dark)

    @property
    def is_dark(self) -> bool:
        return self._dark

    @property
    def colors(self) -> Colors:
        return DARK if self._dark else LIGHT


# --- QSS generation ---------------------------------------------------------

def _combo_arrow_svg(c: Colors) -> str:
    """Write a small chevron SVG for combo down-arrows, return its URL.

    QSS cannot draw a border-triangle on ::down-arrow (Qt renders it as a
    solid block), so we materialize a 12px chevron SVG tinted for the active
    theme and reference it by file URL.
    """
    import tempfile

    # luminance of the window background decides dark vs light
    bg = c.bg.lstrip("#")
    r, g, b = (int(bg[i:i + 2], 16) for i in (0, 2, 4))
    dark = (0.299 * r + 0.587 * g + 0.114 * b) < 128

    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' "
        "viewBox='0 0 12 12'><path d='M2.5 4.25 L6 8 L9.5 4.25' fill='none' "
        f"stroke='{c.text_muted}' stroke-width='1.7' "
        "stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )
    d = Path(tempfile.gettempdir()) / "sharp3d_gui"
    d.mkdir(exist_ok=True)
    f = d / f"combo_arrow_{'dark' if dark else 'light'}.svg"
    f.write_text(svg, encoding="utf-8")
    return f.as_posix()


def build_qss(c: Colors) -> str:
    """Build the full stylesheet for a color set."""

    grad = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c.red}, stop:1 {c.cyan})"
    combo_arrow = _combo_arrow_svg(c)

    return f"""
/* ---------- base ---------- */
QMainWindow, QDialog {{
    background: {c.bg};
}}
QWidget {{
    color: {c.text};
    font-family: "{BODY_FONT}";
    font-size: 13px;
}}
QWidget:disabled {{
    color: {c.text_faint};
}}

/* ---------- tabs ---------- */
QTabWidget::pane {{
    border: 1px solid {c.border};
    border-radius: 10px;
    background: {c.bg};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {c.text_muted};
    padding: 10px 22px 9px 22px;
    margin-right: 4px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 9px;
    border-top-right-radius: 9px;
    font-family: "{DISPLAY_FONT}";
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QTabBar::tab:hover {{
    color: {c.text};
    background: {c.card};
}}
QTabBar::tab:selected {{
    color: {c.text};
    background: {c.bg};
    border-color: {c.border};
}}

/* ---------- cards ---------- */
QFrame[frameShape="6"] {{ /* StyledPanel used for cards */
    background: {c.card};
    border: 1px solid {c.border};
    border-radius: 12px;
}}

/* ---------- buttons ---------- */
QPushButton {{
    background: {c.raised};
    border: 1px solid {c.border};
    border-radius: 8px;
    padding: 7px 16px;
    color: {c.text};
    font-weight: 600;
}}
QPushButton:hover {{
    background: {c.hover};
    border-color: {c.border_strong};
}}
QPushButton:pressed {{
    background: {c.track};
}}
QPushButton:disabled {{
    background: {c.card};
    border-color: {c.border};
}}
QPushButton[cssClass="primary"] {{
    background: {grad};
    border: none;
    color: #0e1013;
    font-family: "{DISPLAY_FONT}";
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 10px 26px;
    border-radius: 9px;
}}
QPushButton[cssClass="primary"]:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c.red_bright}, stop:1 {c.cyan_bright});
}}
QPushButton[cssClass="primary"]:pressed {{
    background: {grad};
}}
QPushButton[cssClass="primary"]:disabled {{
    background: {c.track};
    color: {c.text_faint};
}}
QPushButton[cssClass="danger"] {{
    background: {c.danger};
    border: none;
    color: #ffffff;
    font-weight: 700;
}}
QPushButton[cssClass="danger"]:hover {{
    background: {c.red_bright};
}}

/* ---------- inputs ---------- */
QLineEdit {{
    background: {c.raised};
    border: 1px solid {c.border};
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: {c.cyan};
    selection-color: #0e1013;
}}
QLineEdit:focus {{
    border-color: {c.cyan};
}}
QLineEdit[readOnly="true"] {{
    background: {c.card};
    color: {c.text_muted};
}}

QComboBox {{
    background: {c.raised};
    border: 1px solid {c.border};
    border-radius: 8px;
    padding: 6px 10px;
    min-width: 90px;
}}
QComboBox:hover {{
    border-color: {c.border_strong};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: url("{combo_arrow}");
    width: 12px;
    height: 12px;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {c.card};
    border: 1px solid {c.border_strong};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {c.hover};
    selection-color: {c.text};
    outline: none;
}}

/* ---------- sliders ---------- */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    border-radius: 3px;
    background: {c.track};
}}
QSlider::groove:horizontal[cssClass="stereo"] {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {c.red}, stop:0.5 {c.track}, stop:1 {c.cyan});
}}
QSlider::handle:horizontal {{
    background: {c.card};
    border: 2px solid {c.cyan};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 10px;
}}
QSlider::handle:horizontal:hover {{
    border-color: {c.cyan_bright};
    background: {c.hover};
}}
QSlider::handle:horizontal[cssClass="red"] {{
    border-color: {c.red};
}}
QSlider::handle:horizontal[red="true"] {{
    border-color: {c.red};
}}
QSlider::handle:horizontal[red="true"]:hover {{
    border-color: {c.red_bright};
}}

/* ---------- checkbox ---------- */
QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border: 1.5px solid {c.border_strong};
    border-radius: 5px;
    background: {c.raised};
}}
QCheckBox::indicator:hover {{
    border-color: {c.cyan};
}}
QCheckBox::indicator:checked {{
    background: {c.cyan};
    border-color: {c.cyan};
    image: none;
}}

/* ---------- radio ---------- */
QRadioButton {{
    spacing: 7px;
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1.5px solid {c.border_strong};
    border-radius: 9px;
    background: {c.raised};
}}
QRadioButton::indicator:checked {{
    border: 5px solid {c.cyan};
    background: {c.card};
}}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c.border_strong};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {c.text_faint};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: {c.border_strong};
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------- status bar ---------- */
QStatusBar {{
    background: {c.bg_alt};
    border-top: 1px solid {c.border};
    color: {c.text_muted};
    font-family: "{MONO_FONT}";
    font-size: 11px;
}}
QStatusBar QLabel {{
    color: {c.text_muted};
    font-family: "{MONO_FONT}";
    font-size: 11px;
}}

/* ---------- tooltips ---------- */
QToolTip {{
    background: {c.card};
    color: {c.text};
    border: 1px solid {c.border_strong};
    border-radius: 6px;
    padding: 5px 8px;
}}

/* ---------- group/section titles ---------- */
QLabel[cssClass="cardTitle"] {{
    font-family: "{DISPLAY_FONT}";
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.6px;
    color: {c.text_muted};
    text-transform: uppercase;
}}
QLabel[cssClass="bigNum"] {{
    font-family: "{DISPLAY_FONT}";
    font-weight: 600;
}}
QLabel[cssClass="mono"] {{
    font-family: "{MONO_FONT}";
    font-size: 11px;
    color: {c.text_muted};
}}
QLabel[cssClass="hint"] {{
    color: {c.text_faint};
    font-size: 11px;
}}
"""


def build_palette(c: Colors) -> QPalette:
    """Build a QPalette matching the stylesheet (for native widgets)."""
    p = QPalette()
    p.setColor(QPalette.Window, QColor(c.bg))
    p.setColor(QPalette.WindowText, QColor(c.text))
    p.setColor(QPalette.Base, QColor(c.card))
    p.setColor(QPalette.AlternateBase, QColor(c.raised))
    p.setColor(QPalette.Text, QColor(c.text))
    p.setColor(QPalette.Button, QColor(c.raised))
    p.setColor(QPalette.ButtonText, QColor(c.text))
    p.setColor(QPalette.Highlight, QColor(c.cyan))
    p.setColor(QPalette.HighlightedText, QColor("#0e1013"))
    p.setColor(QPalette.ToolTipBase, QColor(c.card))
    p.setColor(QPalette.ToolTipText, QColor(c.text))
    p.setColor(QPalette.PlaceholderText, QColor(c.text_faint))
    return p


def display_font(size: int, weight: int = QFont.Weight.DemiBold) -> QFont:
    f = QFont(DISPLAY_FONT, size, weight)
    f.setLetterSpacing(QFont.PercentageSpacing, 102)
    return f


def mono_font(size: int = 9) -> QFont:
    return QFont(MONO_FONT, size)
