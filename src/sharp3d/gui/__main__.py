"""Entry point: python -m sharp3d.gui"""

import os
os.environ["PYTHONUTF8"] = "1"  # 修复中文 Windows torch.compile GBK 编码错误

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("sharp3d")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
