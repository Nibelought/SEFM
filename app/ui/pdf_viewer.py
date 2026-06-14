from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtPdf import QPdfDocument, QPdfSearchModel
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui import theme

_WHITE_SCROLL_SS = (
    "QScrollArea { background: white; border: none; }"
    " QScrollArea > QWidget { background: white; }"
)
_NAV_BAR_SS = (
    f"#pdfNav {{ background: {theme.SURFACE};"
    f" border-bottom: 1px solid {theme.BORDER}; }}"
)

# Approx height (px) of the Prev/Next nav bar in normal view.
_NAV_BAR_H = 42


class PdfViewerDialog(QDialog):
    """Modal-less PDF viewer that opens at the cited page.

    Pages render onto an explicit white pixmap so the app chrome never bleeds
    into the PDF. Pass highlight_text to add yellow highlights on the cited page.
    The window is sized to the target page's aspect ratio.
    """

    _RENDER_SCALE = 2.0  # 144 DPI (2x the 72 DPI PDF unit)

    def __init__(
        self,
        pdf_path: Path,
        page: int,
        highlight_text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{pdf_path.name} — p.{page}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        self._doc = QPdfDocument(self)
        self._doc.load(str(pdf_path))
        self._target_page = max(0, min(self._doc.pageCount() - 1, page - 1))
        self._current_page = self._target_page

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if highlight_text:
            self._build_highlighted_view(layout, highlight_text)
            self._size_to_page(extra_height=0)
        else:
            self._build_normal_view(layout)
            self._size_to_page(extra_height=_NAV_BAR_H)

    # ------------------------------------------------------------------
    # Window sizing
    # ------------------------------------------------------------------

    def _size_to_page(self, extra_height: int = 0) -> None:
        """Resize the window to the target page's aspect ratio, clamped to screen."""
        page_pts = self._doc.pagePointSize(self._target_page)
        if page_pts.isEmpty():
            self.resize(900, 1000)
            return

        aspect = page_pts.width() / page_pts.height()
        avail = QApplication.primaryScreen().availableGeometry()
        max_w = int(avail.width() * 0.88)
        max_content_h = int(avail.height() * 0.88) - extra_height

        if max_content_h <= 0:
            max_content_h = 600

        if max_w / max_content_h >= aspect:
            # Height-constrained: fill available height
            content_w = int(max_content_h * aspect)
            content_h = max_content_h
        else:
            # Width-constrained: fill available width
            content_w = max_w
            content_h = int(max_w / aspect)

        self.resize(max(400, content_w), max(400, content_h + extra_height))

    # ------------------------------------------------------------------
    # Normal view (single page, Prev/Next)
    # ------------------------------------------------------------------

    def _build_normal_view(self, layout: QVBoxLayout) -> None:
        nav_bar = QWidget()
        nav_bar.setObjectName("pdfNav")
        nav_bar.setStyleSheet(_NAV_BAR_SS)
        nav = QHBoxLayout(nav_bar)
        nav.setContentsMargins(8, 6, 8, 6)
        self._prev_btn = QPushButton("◀  Prev")
        self._prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next_btn = QPushButton("Next  ▶")
        self._next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._page_lbl = QLabel()
        self._page_lbl.setProperty("muted", True)
        self._page_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self._prev_btn)
        nav.addStretch()
        nav.addWidget(self._page_lbl)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        layout.addWidget(nav_bar)

        self._page_img = QLabel()
        self._page_img.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        self._page_scroll = QScrollArea()
        self._page_scroll.setWidget(self._page_img)
        self._page_scroll.setWidgetResizable(False)
        self._page_scroll.setStyleSheet(_WHITE_SCROLL_SS)
        layout.addWidget(self._page_scroll)

        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)
        self._show_page(self._target_page)

    def _render_to_white_pixmap(self, page_idx: int) -> QPixmap:
        """Render a PDF page to a QPixmap with an explicit white background."""
        page_size = self._doc.pagePointSize(page_idx)
        w = max(1, int(page_size.width() * self._RENDER_SCALE))
        h = max(1, int(page_size.height() * self._RENDER_SCALE))
        img = self._doc.render(page_idx, QSize(w, h))
        pix = QPixmap(w, h)
        pix.fill(QColor(Qt.GlobalColor.white))
        painter = QPainter(pix)
        painter.drawImage(0, 0, img)
        painter.end()
        return pix

    def _show_page(self, page_idx: int) -> None:
        self._current_page = page_idx
        pix = self._render_to_white_pixmap(page_idx)
        self._page_img.setPixmap(pix)
        self._page_img.setFixedSize(pix.size())
        count = self._doc.pageCount()
        self._page_lbl.setText(f"Page {page_idx + 1} of {count}")
        self._prev_btn.setEnabled(page_idx > 0)
        self._next_btn.setEnabled(page_idx < count - 1)
        self._page_scroll.verticalScrollBar().setValue(0)

    def _go_prev(self) -> None:
        if self._current_page > 0:
            self._show_page(self._current_page - 1)

    def _go_next(self) -> None:
        if self._current_page < self._doc.pageCount() - 1:
            self._show_page(self._current_page + 1)

    # ------------------------------------------------------------------
    # Highlighted single-page view
    # ------------------------------------------------------------------

    def _build_highlighted_view(self, layout: QVBoxLayout, search_text: str) -> None:
        self._base_pixmap = self._render_to_white_pixmap(self._target_page)

        self._img_label = QLabel()
        self._img_label.setPixmap(self._base_pixmap)
        self._img_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )

        scroll = QScrollArea()
        scroll.setWidget(self._img_label)
        scroll.setWidgetResizable(False)
        scroll.setStyleSheet(_WHITE_SCROLL_SS)
        layout.addWidget(scroll)

        self._search_model = QPdfSearchModel(self)
        self._search_model.setDocument(self._doc)
        self._search_model.rowsInserted.connect(self._paint_highlights)
        self._search_model.setSearchString(search_text)
        QTimer.singleShot(400, self._paint_highlights)

    def _paint_highlights(self) -> None:
        if not hasattr(self, "_base_pixmap"):
            return
        highlighted = self._base_pixmap.copy()
        painter = QPainter(highlighted)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(255, 220, 0, 140))
        painter.setPen(Qt.PenStyle.NoPen)
        s = self._RENDER_SCALE
        for link in self._search_model.resultsOnPage(self._target_page):
            for rect in link.rectangles():
                painter.drawRect(
                    QRectF(rect.x() * s, rect.y() * s, rect.width() * s, rect.height() * s)
                )
        painter.end()
        self._img_label.setPixmap(highlighted)
