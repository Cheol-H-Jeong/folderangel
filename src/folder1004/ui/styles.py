"""Apple-inspired QSS.

Light and dark variants selected by ``appearance`` config:
  auto  → follow OS (best-effort; Qt 6 exposes color scheme via styleHints)
  light → force light
  dark  → force dark
"""
from __future__ import annotations

LIGHT_QSS = """
* {
    font-family: -apple-system, "SF Pro Text", "Pretendard", "Apple SD Gothic Neo",
                 "Segoe UI", "Yu Gothic UI", "Malgun Gothic",
                 "Noto Sans CJK KR", "Noto Sans KR", sans-serif;
    color: #1d1d1f;
}

QMainWindow, QWidget#MainRoot {
    background: #f5f5f7;
}

QFrame#Sidebar {
    background: #efeff2;
    border-right: 1px solid #e0e0e5;
}

QPushButton#NavItem {
    text-align: left;
    padding: 10px 16px;
    border: none;
    border-radius: 8px;
    background: transparent;
    color: #1d1d1f;
    font-size: 14px;
}
QPushButton#NavItem:hover { background: #e5e5ea; }
QPushButton#NavItem:checked {
    background: #d9d9e0;
    color: #0071e3;
    font-weight: 600;
}

QLabel#Title {
    font-size: 26px;
    font-weight: 700;
    color: #1d1d1f;
}

QLabel#Subtitle {
    font-size: 14px;
    color: #6e6e73;
}

QFrame.Card {
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e4e4e8;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    padding: 8px 10px;
    selection-background-color: #cde3ff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #0071e3;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 10px;
    padding: 8px 14px;
    color: #1d1d1f;
}
QPushButton:hover { background: #f0f0f3; }
QPushButton:disabled { color: #bbbbc0; background: #f5f5f7; }

QPushButton#Primary {
    background: #0071e3;
    color: #ffffff;
    border: 1px solid #0071e3;
    padding: 12px 20px;
    font-weight: 600;
    font-size: 15px;
    border-radius: 12px;
}
QPushButton#Primary:hover { background: #0077ed; }
QPushButton#Primary:disabled { background: #a8c9f2; border-color: #a8c9f2; }

QPushButton#Ghost {
    background: transparent;
    border: 1px solid #d2d2d7;
    border-radius: 10px;
    padding: 8px 14px;
}
QPushButton#Ghost:hover { background: rgba(0,0,0,0.05); }

QPushButton#Warning {
    background: #ff9500;
    color: #ffffff;
    border: 1px solid #ff9500;
    border-radius: 10px;
    padding: 8px 14px;
}

QProgressBar {
    background: #ececf1;
    border: none;
    border-radius: 6px;
    height: 8px;
    text-align: center;
    color: #1d1d1f;
}
QProgressBar::chunk {
    background: #0071e3;
    border-radius: 6px;
}

QTableView, QListView, QTreeView {
    background: #ffffff;
    alternate-background-color: #fafafc;
    selection-background-color: #cde3ff;
    selection-color: #1d1d1f;
    border: 1px solid #e4e4e8;
    border-radius: 10px;
}
QHeaderView::section {
    background: #f5f5f7;
    color: #6e6e73;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #e0e0e5;
    font-weight: 600;
}

QCheckBox { spacing: 8px; }

QStatusBar { background: #efeff2; color: #6e6e73; }

QLabel#Badge {
    background: #e5f1ff;
    color: #0071e3;
    border-radius: 10px;
    padding: 2px 10px;
    font-weight: 600;
}

QLabel#BadgeWarn {
    background: #fff4e5;
    color: #c37200;
    border-radius: 10px;
    padding: 2px 10px;
    font-weight: 600;
}

QTextEdit {
    background: #ffffff;
    border: 1px solid #e4e4e8;
    border-radius: 10px;
    padding: 10px;
}
"""

DARK_QSS = """
* {
    font-family: -apple-system, "SF Pro Text", "Pretendard", "Apple SD Gothic Neo",
                 "Segoe UI", "Yu Gothic UI", "Malgun Gothic",
                 "Noto Sans CJK KR", "Noto Sans KR", sans-serif;
    color: #f2f2f7;
}

QMainWindow, QWidget#MainRoot { background: #1c1c1e; }
QFrame#Sidebar { background: #232326; border-right: 1px solid #2c2c30; }

QPushButton#NavItem {
    text-align: left; padding: 10px 16px; border: none;
    border-radius: 8px; background: transparent; color: #f2f2f7; font-size: 14px;
}
QPushButton#NavItem:hover { background: #2c2c30; }
QPushButton#NavItem:checked {
    background: #303036;
    color: #0a84ff;
    font-weight: 600;
}

QLabel#Title { font-size: 26px; font-weight: 700; color: #f2f2f7; }
QLabel#Subtitle { font-size: 14px; color: #a1a1a6; }

QFrame.Card {
    background: #232326; border-radius: 14px; border: 1px solid #2c2c30;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background: #2c2c30; border: 1px solid #3a3a3d; color: #f2f2f7;
    border-radius: 8px; padding: 8px 10px;
    selection-background-color: #264f78;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #0a84ff;
}

QPushButton {
    background: #2c2c30; border: 1px solid #3a3a3d; border-radius: 10px;
    padding: 8px 14px; color: #f2f2f7;
}
QPushButton:hover { background: #34343a; }
QPushButton:disabled { color: #6e6e73; background: #2a2a2f; }

QPushButton#Primary {
    background: #0a84ff; color: #ffffff; border: 1px solid #0a84ff;
    padding: 12px 20px; font-weight: 600; font-size: 15px; border-radius: 12px;
}
QPushButton#Primary:hover { background: #1a8fff; }
QPushButton#Primary:disabled { background: #294b74; border-color: #294b74; }

QPushButton#Ghost { background: transparent; border: 1px solid #3a3a3d; }

QProgressBar {
    background: #2c2c30; border: none; border-radius: 6px; height: 8px;
    text-align: center; color: #f2f2f7;
}
QProgressBar::chunk { background: #0a84ff; border-radius: 6px; }

QTableView, QListView, QTreeView {
    background: #232326; alternate-background-color: #26262a;
    selection-background-color: #264f78; selection-color: #ffffff;
    border: 1px solid #2c2c30; border-radius: 10px;
}
QHeaderView::section {
    background: #1c1c1e; color: #a1a1a6; padding: 6px;
    border: none; border-bottom: 1px solid #2c2c30; font-weight: 600;
}

QStatusBar { background: #232326; color: #a1a1a6; }

QLabel#Badge { background: #1e3a5a; color: #64b5ff; border-radius: 10px; padding: 2px 10px; font-weight: 600; }
QLabel#BadgeWarn { background: #4a3300; color: #ffb547; border-radius: 10px; padding: 2px 10px; font-weight: 600; }
"""


def resolve_qss(appearance: str) -> str:
    appearance = (appearance or "auto").lower()
    if appearance == "dark":
        return DARK_QSS
    if appearance == "light":
        return LIGHT_QSS
    # auto: consult OS
    try:
        from PySide6.QtGui import QGuiApplication  # type: ignore
        from PySide6.QtCore import Qt  # type: ignore

        hints = QGuiApplication.styleHints()
        if hints and hasattr(hints, "colorScheme"):
            scheme = hints.colorScheme()
            if scheme == Qt.ColorScheme.Dark:
                return DARK_QSS
    except Exception:
        pass
    return LIGHT_QSS
