from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Palette — a single source of truth for colours used in QSS and in code.
# Light, neutral chrome with a blue accent; matches the white PDF pages so the
# viewer no longer has to fight a dark palette.
# ---------------------------------------------------------------------------
BG = "#f4f6f9"           # window background
SURFACE = "#ffffff"      # cards, inputs, lists
SURFACE_ALT = "#eef1f5"  # headers, hovered rows
BORDER = "#d6dbe3"       # default 1px borders
BORDER_STRONG = "#c2c9d4"
TEXT = "#1f2328"
TEXT_MUTED = "#6b7280"
ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
ACCENT_PRESSED = "#1e40af"
ACCENT_DISABLED = "#a8c0f0"
SUCCESS = "#16a34a"
WARNING = "#b45309"
DANGER = "#dc2626"
SELECTION_BG = "#2563eb"
SELECTION_FG = "#ffffff"

# QSS for the whole application. Sub-control arrows on combos/spinboxes are left
# to Fusion so they never render blank.
STYLESHEET = f"""
QMainWindow, QDialog {{
    background: {BG};
}}
QWidget {{
    color: {TEXT};
}}

/* --- Buttons --------------------------------------------------------- */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {SURFACE_ALT};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {SURFACE_ALT};
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    background: {SURFACE_ALT};
    border-color: {BORDER};
}}
QPushButton[accent="true"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[accent="true"]:pressed {{
    background: {ACCENT_PRESSED};
}}
QPushButton[accent="true"]:disabled {{
    background: {ACCENT_DISABLED};
    border-color: {ACCENT_DISABLED};
    color: #eef2ff;
}}

/* --- Text inputs ----------------------------------------------------- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {SELECTION_BG};
    selection-color: {SELECTION_FG};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QTextEdit:disabled {{
    background: {SURFACE_ALT};
    color: {TEXT_MUTED};
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {SELECTION_BG};
    selection-color: {SELECTION_FG};
    outline: none;
}}

/* --- Lists / trees --------------------------------------------------- */
QTreeWidget, QTreeView, QListView, QTableView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    alternate-background-color: {SURFACE_ALT};
    outline: none;
}}
QTreeWidget::item, QTreeView::item {{
    padding: 4px 2px;
    border: none;
}}
QTreeWidget::item:selected, QTreeView::item:selected {{
    background: {SELECTION_BG};
    color: {SELECTION_FG};
}}
QHeaderView::section {{
    background: {SURFACE_ALT};
    color: {TEXT_MUTED};
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}

/* --- Tabs ------------------------------------------------------------ */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    top: -1px;
    background: {BG};
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 8px 18px;
    margin-right: 2px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{
    color: {TEXT};
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}

/* --- Group boxes ----------------------------------------------------- */
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
}}

/* --- Progress bar ---------------------------------------------------- */
QProgressBar {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    height: 14px;
    text-align: center;
    color: {TEXT};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 6px;
}}

/* --- Scroll bars ----------------------------------------------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    min-height: 28px;
    border-radius: 6px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_MUTED}; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    min-width: 28px;
    border-radius: 6px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: {TEXT_MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* --- Menu / status bars --------------------------------------------- */
QMenuBar {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item {{
    padding: 6px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background: {SURFACE_ALT};
    border-radius: 4px;
}}
QMenu {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {SELECTION_BG};
    color: {SELECTION_FG};
}}
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QStatusBar::item {{ border: none; }}

/* --- Tooltips -------------------------------------------------------- */
QToolTip {{
    background: {TEXT};
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
}}

/* --- Tool buttons (e.g. the Settings gear in the tab corner) --------- */
QToolButton {{
    border: none;
    background: transparent;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 15px;
    color: {TEXT_MUTED};
}}
QToolButton:hover {{
    background: {SURFACE_ALT};
    color: {TEXT};
}}
QToolButton:pressed {{
    background: {BORDER};
}}

/* --- Citation chips -------------------------------------------------- */
QFrame#citationChip {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
}}
QFrame#citationChip:hover {{
    border-color: {ACCENT};
}}
QFrame#citationChip QLabel {{
    color: {TEXT};
    background: transparent;
}}
QPushButton#chipOpen, QPushButton#chipVerify {{
    border: none;
    background: transparent;
    padding: 2px 8px;
    border-radius: 9px;
    min-height: 0;
    font-weight: 600;
}}
QPushButton#chipOpen {{ color: {ACCENT}; }}
QPushButton#chipVerify {{ color: {TEXT_MUTED}; font-weight: 500; }}
QPushButton#chipOpen:hover, QPushButton#chipVerify:hover {{
    background: {SURFACE_ALT};
}}

/* --- Semantic label roles (set via dynamic properties) --------------- */
QLabel[heading="true"] {{
    font-size: 12px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel[muted="true"] {{
    color: {TEXT_MUTED};
}}
"""


def app_icon() -> QIcon:
    """A simple drawn app icon: an accent rounded square with a white 'S'."""
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(ACCENT))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(4, 4, 56, 56), 14, 14)
    p.setPen(QColor("#ffffff"))
    font = QFont("Segoe UI", 30, QFont.Weight.Bold)
    p.setFont(font)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "S")
    p.end()
    return QIcon(pix)


def apply_theme(app: QApplication) -> None:
    """Apply SEFM's look (Fusion base + QSS + app font + icon) to the app."""
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(app_icon())
