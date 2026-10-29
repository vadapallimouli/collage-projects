"""
gui_components.py
=================

Reusable styling helpers, widgets and the background-task worker used by
``main.py``.  This module contains NO project/ML logic and never imports
``main.py``.

Contents
--------
* ``Colors`` / ``build_stylesheet()`` - the white / pale-lavender / navy /
  violet visual theme as a Qt stylesheet (no external image assets).
* Small widgets: ``DropArea``, ``PlotPanel``, ``ScaledImageLabel``,
  ``MetricCard``, ``ResponsiveGrid``, ``NumericItem``.
* ``TaskWorker`` - runs a callable in a ``QThread`` and reports back through
  Qt signals only (widgets are never touched from the worker thread).
* ``draw_confusion_matrix()`` - pyqtgraph heat-map helper.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("gui_components")

# Row-major images: arr[row, col] is drawn at (x=col, y=row).
pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


# ==============================================================================
# THEME
# ==============================================================================

class Colors:
    """Colour tokens for the application theme."""

    BG = "#FFFFFF"
    SURFACE = "#F7F5FD"          # very pale lavender
    SURFACE_ALT = "#EFECFA"
    BORDER = "#E3DEF4"
    BORDER_STRONG = "#CFC8EA"
    TEXT = "#141B4D"             # deep navy
    MUTED = "#5B6190"
    FAINT = "#8A8FB5"
    ACCENT = "#6B4FD8"           # restrained violet
    ACCENT_DARK = "#5238B8"
    ACCENT_SOFT = "#E9E4FB"
    SUCCESS = "#1B7F63"
    SUCCESS_BG = "#E3F4EE"
    WARN = "#8A5E0C"
    WARN_BG = "#FBF1DC"
    DANGER = "#B3364A"
    DANGER_BG = "#FBE7EA"
    PLOT_A = "#5B43C7"
    PLOT_B = "#2F8F9D"
    PLOT_C = "#8B93B8"


_QSS_TEMPLATE = """
QWidget#root, QWidget#page { background: @BG@; }
QScrollArea { background: transparent; border: none; }
QLabel { background: transparent; color: @TEXT@; }
QLabel#title { font-size: 20pt; font-weight: 700; }
QLabel#subtitle { color: @MUTED@; font-size: 10pt; }
QLabel#sectionTitle { font-size: 14pt; font-weight: 700; }
QLabel#cardTitle { color: @MUTED@; font-size: 9pt; font-weight: 700; }
QLabel#muted { color: @MUTED@; }
QLabel#faint { color: @FAINT@; font-size: 9pt; }
QLabel#bigValue { color: @ACCENT_DARK@; font-size: 30pt; font-weight: 700; }
QLabel#metricValue { color: @TEXT@; font-size: 20pt; font-weight: 700; }
QLabel#featureValue { color: @ACCENT_DARK@; font-size: 15pt; font-weight: 700; }
QLabel#featureName { font-size: 10pt; font-weight: 700; }
QLabel#notice {
    background: @ACCENT_SOFT@; border: 1px solid @BORDER_STRONG@;
    border-radius: 8px; padding: 10px 12px;
}
QLabel#warnNotice {
    background: @WARN_BG@; border: 1px solid #E8D3A0;
    border-radius: 8px; padding: 10px 12px; color: @WARN@;
}
QLabel#emptyState {
    background: @BG@; border: 1px dashed @BORDER_STRONG@;
    border-radius: 10px; color: @MUTED@; padding: 16px;
}
QLabel#pill { border-radius: 10px; padding: 4px 12px; font-weight: 700; font-size: 9pt; }
QLabel#pill[kind="ok"] { background: @SUCCESS_BG@; color: @SUCCESS@; }
QLabel#pill[kind="busy"] { background: @WARN_BG@; color: @WARN@; }
QLabel#pill[kind="error"] { background: @DANGER_BG@; color: @DANGER@; }
QLabel#pill[kind="info"] { background: @ACCENT_SOFT@; color: @ACCENT_DARK@; }

QFrame#header { background: @BG@; border: 1px solid @BORDER@; border-radius: 14px; }
QFrame#card { background: @SURFACE@; border: 1px solid @BORDER@; border-radius: 12px; }
QFrame#metricCard { background: @BG@; border: 1px solid @BORDER@; border-radius: 10px; }
QFrame#dropArea {
    background: @BG@; border: 2px dashed @BORDER_STRONG@; border-radius: 12px;
}
QFrame#dropArea[hover="true"] { border-color: @ACCENT@; background: @ACCENT_SOFT@; }

QPushButton {
    background: @ACCENT@; color: #FFFFFF; border: none; border-radius: 8px;
    padding: 9px 18px; font-weight: 600;
}
QPushButton:hover { background: @ACCENT_DARK@; }
QPushButton:disabled { background: #D9D4EE; color: #FFFFFF; }
QPushButton#secondary {
    background: @BG@; color: @ACCENT_DARK@; border: 1px solid @BORDER_STRONG@;
}
QPushButton#secondary:hover { background: @ACCENT_SOFT@; }
QPushButton#secondary:disabled { background: @BG@; color: #B7B3D3; border: 1px solid @BORDER@; }

QTabWidget::pane { border: none; top: 0px; }
QTabBar::tab {
    background: transparent; color: @MUTED@; padding: 10px 22px; margin-right: 4px;
    border: none; border-bottom: 3px solid transparent; font-weight: 600; font-size: 11pt;
}
QTabBar::tab:selected { color: @ACCENT_DARK@; border-bottom: 3px solid @ACCENT@; }
QTabBar::tab:hover:!selected { color: @TEXT@; }
QTabWidget#subTabs QTabBar::tab {
    font-size: 10pt; padding: 7px 16px; border: none; border-radius: 8px; margin-bottom: 6px;
}
QTabWidget#subTabs QTabBar::tab:selected { background: @ACCENT_SOFT@; color: @ACCENT_DARK@; border: none; }

QTableWidget {
    background: @BG@; alternate-background-color: @SURFACE@; border: 1px solid @BORDER@;
    border-radius: 8px; gridline-color: @BORDER@;
    selection-background-color: @ACCENT_SOFT@; selection-color: @TEXT@;
}
QHeaderView::section {
    background: @SURFACE_ALT@; color: @MUTED@; padding: 7px 8px; border: none;
    border-right: 1px solid @BORDER@; font-weight: 700;
}

QComboBox, QSpinBox, QDoubleSpinBox {
    background: @BG@; border: 1px solid @BORDER_STRONG@; border-radius: 7px;
    padding: 6px 10px; min-height: 20px; color: @TEXT@;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: #A9ADC9; background: @SURFACE@; }
QComboBox QAbstractItemView {
    background: @BG@; color: @TEXT@; border: 1px solid @BORDER@;
    selection-background-color: @ACCENT_SOFT@; selection-color: @TEXT@;
}

QProgressBar {
    background: @SURFACE_ALT@; border: none; border-radius: 5px; text-align: center;
    min-height: 10px; max-height: 10px; color: transparent;
}
QProgressBar::chunk { background: @ACCENT@; border-radius: 5px; }
QProgressBar#statusProgress { max-width: 200px; }

QScrollBar:vertical { background: transparent; width: 11px; margin: 2px; }
QScrollBar::handle:vertical { background: @BORDER_STRONG@; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 2px; }
QScrollBar::handle:horizontal { background: @BORDER_STRONG@; border-radius: 4px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

QStatusBar { background: @SURFACE@; color: @TEXT@; border-top: 1px solid @BORDER@; }
QStatusBar::item { border: none; }
QToolTip { background: @TEXT@; color: #FFFFFF; border: none; padding: 5px 8px; }
"""


def build_stylesheet() -> str:
    """Returns the application stylesheet with colour tokens substituted."""
    qss = _QSS_TEMPLATE
    for name in dir(Colors):
        if name.isupper():
            qss = qss.replace(f"@{name}@", getattr(Colors, name))
    return qss


def repolish(widget: QWidget) -> None:
    """Re-applies the stylesheet after a dynamic property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def set_pill(label: QLabel, text: str, kind: str) -> None:
    """Sets pill text and kind (``ok``, ``busy``, ``error`` or ``info``)."""
    label.setText(text)
    label.setProperty("kind", kind)
    repolish(label)


def add_shadow(widget: QWidget, blur: int = 22, offset_y: int = 3, alpha: int = 24) -> None:
    """Adds a soft drop shadow (use sparingly; avoid on plot-heavy widgets)."""
    from PySide6.QtGui import QColor

    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, offset_y)
    effect.setColor(QColor(40, 30, 110, alpha))
    widget.setGraphicsEffect(effect)


def make_card(title: Optional[str] = None, margins: Tuple[int, int, int, int] = (16, 14, 16, 16),
              spacing: int = 10) -> Tuple[QFrame, QVBoxLayout]:
    """Creates a rounded card frame with an optional small-caps title."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    if title:
        label = QLabel(title.upper())
        label.setObjectName("cardTitle")
        layout.addWidget(label)
    return frame, layout


def style_plot(plot: pg.PlotWidget, x_label: str, y_label: str, title: Optional[str] = None) -> None:
    """Applies the scientific-plot look to a pyqtgraph ``PlotWidget``."""
    plot.setBackground(Colors.BG)
    if title:
        plot.setTitle(title, color=Colors.MUTED, size="10pt")
    plot.setLabel("bottom", x_label, color=Colors.MUTED)
    plot.setLabel("left", y_label, color=Colors.MUTED)
    plot.showGrid(x=True, y=True, alpha=0.18)
    for name in ("bottom", "left"):
        axis = plot.getAxis(name)
        axis.setPen(pg.mkPen(Colors.BORDER_STRONG))
        axis.setTextPen(pg.mkPen(Colors.MUTED))
    plot.setMenuEnabled(False)
    plot.hideButtons()


def fmt_number(value: Any, digits: int = 4) -> str:
    """Formats a number for display (scientific notation for extremes)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    if number == 0.0:
        return "0"
    magnitude = abs(number)
    if magnitude >= 1e6 or magnitude < 1e-3:
        return f"{number:.4e}"
    return f"{number:,.{digits}f}"


# ==============================================================================
# WIDGETS
# ==============================================================================

class PlotPanel(QStackedWidget):
    """A ``PlotWidget`` that shows a helpful empty-state message until used."""

    def __init__(self, x_label: str, y_label: str, empty_text: str,
                 min_height: int = 230, title: Optional[str] = None) -> None:
        super().__init__()
        self.plot = pg.PlotWidget()
        style_plot(self.plot, x_label, y_label, title)
        self._empty = QLabel(empty_text)
        self._empty.setObjectName("emptyState")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self.addWidget(self._empty)
        self.addWidget(self.plot)
        self.setMinimumHeight(min_height)
        self.show_empty()

    def show_empty(self, text: Optional[str] = None) -> None:
        if text is not None:
            self._empty.setText(text)
        self.setCurrentWidget(self._empty)

    def show_plot(self) -> None:
        self.setCurrentWidget(self.plot)


class DropArea(QFrame):
    """Click-or-drop target that accepts a single ``.npy`` file."""

    fileDropped = Signal(str)
    clicked = Signal()

    def __init__(self, title: str, detail: str) -> None:
        super().__init__()
        self.setObjectName("dropArea")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title = QLabel(title)
        self._title.setObjectName("sectionTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail = QLabel(detail)
        self._detail.setObjectName("muted")
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail.setWordWrap(True)
        layout.addWidget(self._title)
        layout.addWidget(self._detail)

    def set_text(self, title: str, detail: str) -> None:
        self._title.setText(title)
        self._detail.setText(detail)

    @staticmethod
    def _first_npy(mime: Any) -> Optional[str]:
        for url in mime.urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".npy"):
                return url.toLocalFile()
        return None

    def _set_hover(self, hover: bool) -> None:
        self.setProperty("hover", "true" if hover else "false")
        repolish(self)

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        if self._first_npy(event.mimeData()):
            event.acceptProposedAction()
            self._set_hover(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event: Any) -> None:  # noqa: N802
        self._set_hover(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        self._set_hover(False)
        path = self._first_npy(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.fileDropped.emit(path)
        else:
            event.ignore()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ScaledImageLabel(QLabel):
    """Shows a PNG scaled to the available width, or a placeholder message."""

    def __init__(self, placeholder: str, max_height: int = 560) -> None:
        super().__init__(placeholder)
        self.setObjectName("emptyState")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._pixmap: Optional[QPixmap] = None
        self._max_height = max_height

    def set_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.set_message(f"Could not load image:\n{path.name}")
            return False
        self._pixmap = pixmap
        self.setToolTip(str(path))
        self._rescale()
        return True

    def set_message(self, text: str) -> None:
        self._pixmap = None
        self.clear()
        self.setText(text)
        self.setMinimumHeight(140)
        self.setMaximumHeight(16777215)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self.width() < 50:
            return
        scaled = self._pixmap.scaledToWidth(max(50, self.width() - 4), Qt.TransformationMode.SmoothTransformation)
        if scaled.height() > self._max_height:
            scaled = scaled.scaledToHeight(self._max_height, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled)
        self.setFixedHeight(scaled.height() + 4)


class MetricCard(QFrame):
    """Compact KPI card: small title, large value, optional note."""

    def __init__(self, title: str, value: str = "-", note: str = "") -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)
        self._title = QLabel(title.upper())
        self._title.setObjectName("cardTitle")
        self._value = QLabel(value)
        self._value.setObjectName("metricValue")
        self._note = QLabel(note)
        self._note.setObjectName("faint")
        layout.addWidget(self._title)
        layout.addWidget(self._value)
        layout.addWidget(self._note)

    def set_value(self, value: str, note: Optional[str] = None) -> None:
        self._value.setText(value)
        if note is not None:
            self._note.setText(note)


class ResponsiveGrid(QWidget):
    """Grid that re-flows its children into more/fewer columns on resize."""

    def __init__(self, min_item_width: int = 300, spacing: int = 12) -> None:
        super().__init__()
        self._min_width = min_item_width
        self._items: List[QWidget] = []
        self._cols = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(spacing)

    def add_widget(self, widget: QWidget) -> None:
        self._items.append(widget)
        self._relayout(force=True)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self, force: bool = False) -> None:
        if not self._items:
            return
        cols = max(1, min(len(self._items), self.width() // self._min_width))
        if cols == self._cols and not force:
            return
        self._cols = cols
        while self._grid.count():
            self._grid.takeAt(0)
        for index, widget in enumerate(self._items):
            self._grid.addWidget(widget, index // cols, index % cols)
        for column in range(12):
            self._grid.setColumnStretch(column, 1 if column < cols else 0)


class NumericItem(QTableWidgetItem):
    """Table item that sorts numerically while showing formatted text."""

    def __init__(self, value: Any, text: Optional[str] = None,
                 align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        super().__init__(text if text is not None else fmt_number(value))
        self._number = number
        self.setTextAlignment(align)
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, NumericItem):
            a, b = self._number, other._number
            if np.isnan(a):
                return not np.isnan(b)
            if np.isnan(b):
                return False
            return a < b
        return super().__lt__(other)


# ==============================================================================
# BACKGROUND WORKER
# ==============================================================================

class TaskWorker(QObject):
    """Runs ``fn(progress_cb, should_cancel)`` on a worker thread.

    ``fn`` must not touch any widget.  Results are delivered through Qt
    signals, which Qt queues to the GUI thread because the receivers
    (bound methods of GUI objects) live there.
    """

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn: Callable[[Callable[[int, int, str], None], Callable[[], bool]], Any]) -> None:
        super().__init__()
        self._fn = fn
        self._cancel_event = threading.Event()
        self._last_emit = 0.0

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _emit_progress(self, done: int, total: int, message: str = "") -> None:
        now = time.monotonic()
        if done >= total or now - self._last_emit >= 0.08:
            self._last_emit = now
            self.progress.emit(int(done), int(total), str(message))

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(self._emit_progress, self.is_cancelled)
        except Exception as exc:  # noqa: BLE001 - reported to the GUI
            if getattr(exc, "is_cancellation", False):
                self.cancelled.emit()
                return
            logger.exception("Background task failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)


# ==============================================================================
# PLOT HELPERS
# ==============================================================================

def draw_confusion_matrix(plot: pg.PlotWidget, matrix: Sequence[Sequence[float]],
                          labels: Sequence[str]) -> None:
    """Draws a labelled confusion-matrix heat-map (rows = true, cols = predicted)."""
    arr = np.asarray(matrix, dtype=float)
    n = arr.shape[0]
    plot.clear()
    image = pg.ImageItem(arr)
    colormap = pg.ColorMap([0.0, 1.0], [(247, 245, 253), (91, 67, 199)])
    image.setLookupTable(colormap.getLookupTable(0.0, 1.0, 256))
    vmax = float(arr.max()) if arr.size and arr.max() > 0 else 1.0
    image.setLevels((0.0, vmax))
    plot.addItem(image)
    for i in range(n):
        for j in range(n):
            value = arr[i, j]
            text = pg.TextItem(str(int(round(value))), color=("#FFFFFF" if value > 0.55 * vmax else Colors.TEXT),
                               anchor=(0.5, 0.5))
            text.setFont(QFont("Segoe UI", 10))
            plot.addItem(text)
            text.setPos(j + 0.5, i + 0.5)
    ticks = [[(k + 0.5, str(labels[k]).replace("_", "-")) for k in range(n)]]
    plot.getAxis("bottom").setTicks(ticks)
    plot.getAxis("left").setTicks(ticks)
    plot.showGrid(x=False, y=False)
    plot.setXRange(0, n, padding=0)
    plot.setYRange(0, n, padding=0)
    plot.invertY(True)
    plot.setMouseEnabled(False, False)