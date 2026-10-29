"""Analog Modulation Classifier - desktop GUI (PySide6 + PyQtGraph).

Place this file beside predict.py, preprocessing.py, feature_extraction.py
and the trained_models/ folder (i.e. replace your existing GUI file).

Workflow
--------
* Tab 1 "Analyze": choose one or more .npy files, click "Analyze signal(s)".
  Choosing files never runs the model.
* Tab 2 "Signal details": pick any successfully analyzed signal to see its
  waveform, frequency spectrum, measurements and a structured explanation.

The ML pipeline (predict.SignalPredictor) is used unchanged. Everything done
for plotting is display-only and never feeds back into the classifier.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFileDialog, QFrame,
    QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent

try:
    from predict import PredictConfig, SignalLoader, SignalPredictor, SAMPLING_FREQUENCY
    BACKEND_ERROR = None
except Exception as exc:  # pragma: no cover - reported in the UI instead
    PredictConfig = SignalPredictor = SignalLoader = None
    SAMPLING_FREQUENCY = 20_000.0
    BACKEND_ERROR = f"{type(exc).__name__}: {exc}"

#: Sampling frequency comes from the project's own predict.py (single source of truth).
FS = float(SAMPLING_FREQUENCY)

PURPLE = "#6847F5"
TEAL = "#168BA0"
INK = "#242044"
MUTED = "#737D95"
MIN_DISPLAY_SAMPLES = 8


# ==============================================================================
# Pure, Qt-free helpers (display only - nothing here touches the model input)
# ==============================================================================

def prepare_signal_view(signal: Any, fs: float) -> dict:
    """Validate a raw signal and compute everything the Details tab needs.

    The input array is never modified. All statistics are computed on a copy
    scaled by the peak amplitude and then rescaled, so very large amplitudes
    (e.g. 1e200) cannot overflow float64 when squared.

    Raises:
        ValueError: with a user-readable message if the signal cannot be shown.
    """
    arr = np.asarray(signal)
    if arr.size == 0:
        raise ValueError("The signal is empty (zero samples).")
    if np.iscomplexobj(arr) or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"The signal must be real-valued numeric data (dtype={arr.dtype}).")
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D signal, got an array of shape {arr.shape}.")

    x = arr.astype(np.float64, copy=True)
    bad = ~np.isfinite(x)
    if bad.any():
        raise ValueError(f"The signal contains {int(bad.sum())} NaN or infinite sample(s).")
    n = int(x.size)
    if n < MIN_DISPLAY_SAMPLES:
        raise ValueError(f"The signal has only {n} samples; at least {MIN_DISPLAY_SAMPLES} are needed.")

    peak = float(np.max(np.abs(x)))
    u = x / peak if peak > 0.0 else np.zeros_like(x)     # normalized copy, |u| <= 1

    mean_u = float(np.mean(u))
    rms_u = float(np.sqrt(np.mean(u * u)))
    std_u = float(np.std(u))

    # Spectrum: remove DC, Hann window, single-sided amplitude spectrum.
    window = np.hanning(n)
    wsum = float(np.sum(window))
    mag = np.abs(np.fft.rfft((u - mean_u) * window)) / max(wsum, 1e-12) * 2.0
    mag[0] *= 0.5
    if n % 2 == 0 and mag.size > 1:
        mag[-1] *= 0.5
    freq_khz = np.fft.rfftfreq(n, d=1.0 / fs) / 1000.0
    mag_max = float(np.max(mag))

    return {
        "n": n,
        "fs": float(fs),
        "duration_s": n / fs,
        "mean": peak * mean_u,
        "rms": peak * rms_u,
        "peak": peak,
        "std": peak * std_u,
        "crest": (1.0 / rms_u) if rms_u > 1e-12 else None,
        "time_ms": np.arange(n, dtype=np.float64) / fs * 1000.0,
        "display": u,
        "freq_khz": freq_khz,
        "magnitude": mag,
        "mag_ymax": mag_max * 1.1 if mag_max > 1e-12 else 1.0,
        "nyquist_khz": fs / 2000.0,
        "all_zero": peak == 0.0,
    }


def fmt_num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def fmt_pct(value: Optional[float]) -> str:
    return "Unavailable" if value is None else f"{value * 100:.1f}%"


def ranked_scores(probabilities: Any) -> list[tuple[str, float]]:
    """Return [(class, score), ...] sorted high to low from a dict or list of pairs."""
    if not probabilities:
        return []
    items = probabilities.items() if isinstance(probabilities, dict) else probabilities
    return sorted(((str(k).replace("_", "-"), float(v)) for k, v in items),
                  key=lambda item: item[1], reverse=True)


CLASS_MEANING = {
    "AM": "The carrier's <b>amplitude</b> changes with the message signal. The waveform "
          "envelope rises and falls, and the spectrum usually shows a carrier peak with "
          "sidebands on both sides.",
    "FM": "The carrier's <b>instantaneous frequency</b> changes with the message signal, "
          "while the amplitude stays roughly constant. The spectrum tends to spread over a "
          "wider band with several side-frequency peaks.",
    "DSB-SC": "The message modulates the carrier, and the <b>carrier component is ideally "
              "suppressed</b>. Two sidebands remain, one above and one below the carrier "
              "frequency, with little or no energy at the carrier itself.",
    "SSB": "<b>Only one sideband</b> is transmitted (the other sideband and ideally the "
           "carrier are removed), so the signal occupies roughly half the bandwidth of a "
           "double-sideband signal.",
}


# ==============================================================================
# Widgets that must never swallow the mouse wheel (so the page always scrolls)
# ==============================================================================

class StaticPlot(pg.PlotWidget):
    """Read-only plot: no mouse zoom/pan/menu, and wheel events scroll the page."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        item = self.getPlotItem()
        vb = item.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)
        item.setMenuEnabled(False)
        item.hideButtons()
        self.setBackground("#FFFFFF")

    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class StaticTable(QTableWidget):
    def wheelEvent(self, event):
        event.ignore()


# ==============================================================================
# Background worker (existing structure preserved)
# ==============================================================================

class PredictionWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, paths: list[Path], predictor: Any):
        super().__init__()
        self.paths = paths
        self.predictor = predictor

    def run(self):
        rows = []
        try:
            total = len(self.paths)
            for i, path in enumerate(self.paths, 1):
                rows.append(self._analyze_one(path))
                self.progress.emit(i, total)
            self.finished.emit(rows)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _analyze_one(self, path: Path) -> dict:
        try:
            result = self.predictor.predict(path)
            ranked = ranked_scores(result.class_probabilities)
            confidence = (
                float(result.confidence) if result.confidence is not None
                else (ranked[0][1] if ranked else None)
            )
            predicted = str(result.predicted_modulation).replace("_", "-")
            reason = (
                f"Top model score: {predicted} ({confidence * 100:.1f}%)."
                if confidence is not None
                else "The model selected this class from the extracted signal features."
            )
            if len(ranked) > 1:
                reason += f" Next: {ranked[1][0]} ({ranked[1][1] * 100:.1f}%)."
            row = {
                "path": path, "predicted": predicted, "confidence": confidence,
                "why": reason, "probabilities": ranked, "features": result.features,
                "model_name": getattr(result, "model_name", None),
                "error": "", "view": None, "view_error": "",
            }
            # Display data is prepared once, here, from the file that was analyzed,
            # so the Details tab can never show a different signal's data.
            try:
                raw = SignalLoader().load(path) if SignalLoader else np.load(path, allow_pickle=False)
                row["view"] = prepare_signal_view(raw, FS)
            except Exception as exc:
                row["view_error"] = str(exc)
            return row
        except Exception as exc:
            return {
                "path": path, "predicted": "Could not analyze", "confidence": None,
                "why": str(exc), "probabilities": [], "features": {},
                "model_name": None, "error": str(exc), "view": None, "view_error": "",
            }


# ==============================================================================
# Main window
# ==============================================================================

STYLE = """
QWidget { background: #F5F6FA; color: #172033; font-family: "Segoe UI"; font-size: 10pt; }
QLabel { background: transparent; border: none; }
#root, #pageBody { background: #F5F6FA; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #F0F1F6; width: 12px; margin: 3px; border-radius: 6px; }
QScrollBar::handle:vertical { background: #B8BED0; min-height: 34px; border-radius: 6px; }
QScrollBar::handle:vertical:hover { background: #929BB4; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background: #F0F1F6; height: 12px; margin: 3px; border-radius: 6px; }
QScrollBar::handle:horizontal { background: #B8BED0; min-width: 34px; border-radius: 6px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

#header { background: #FFFFFF; border: 1px solid #E3E7F0; border-radius: 16px; }
#card { background: #FFFFFF; border: 1px solid #E3E7F0; border-radius: 16px; }
#section { background: #FFFFFF; border: 1px solid #E3E7F0; border-radius: 14px; }
#metric { background: #F7F8FC; border: 1px solid #E8EBF3; border-radius: 11px; }
#summaryBlock { background: transparent; }

#title { font-size: 20pt; font-weight: 700; color: #18213B; }
#eyebrow { color: #7B849B; font-size: 8pt; font-weight: 700; }
#h1 { font-size: 16pt; font-weight: 700; color: #30295A; }
#h2 { font-size: 12pt; font-weight: 700; color: #30295A; }
#sectionTitle { font-size: 11pt; font-weight: 700; color: #242044; }
#cardTitle { font-size: 13pt; font-weight: 700; color: #222D49; }
#caption { color: #7B849B; font-size: 8.5pt; }
#muted { color: #737D95; }
#metricLabel { color: #7B849B; font-size: 8.5pt; font-weight: 600; }
#metricValue { color: #242044; font-size: 12pt; font-weight: 700; }
#blockLabel { color: #7B849B; font-size: 8pt; font-weight: 700; }
#blockValue { color: #242044; font-size: 12pt; font-weight: 700; }
#blockValueBig { color: #6847F5; font-size: 17pt; font-weight: 700; }
#emptyTitle { color: #30295A; font-size: 13pt; font-weight: 700; }

#status { background: #E8F7EF; color: #14764E; border: 1px solid #CDEEDD;
          border-radius: 11px; padding: 9px 15px; font-weight: 600; }
#status[state="busy"] { background: #F0EDFF; color: #6847F5; border-color: #E1DBFF; }
#status[state="error"] { background: #FDECEC; color: #A52A2A; border-color: #F6CFCF; }

#uploadIcon { background: #F0EDFF; color: #6847F5; border-radius: 15px; font-size: 22pt; font-weight: 600; }

#activity { background: #F0EDFF; border: 1px solid #E1DBFF; border-radius: 10px;
            padding: 10px 14px; color: #6847F5; font-weight: 700; }
#activity[state="success"] { background: #E8F7EF; border-color: #CDEEDD; color: #14764E; }
#activity[state="warn"] { background: #FFF7E8; border-color: #F5DFB0; color: #865A12; }
#activity[state="error"] { background: #FDECEC; border-color: #F6CFCF; color: #A52A2A; }
#hint { background: #F0EDFF; border: 1px solid #E1DBFF; border-radius: 11px;
        padding: 12px 15px; color: #51418F; }
#note { background: #FFF7E8; border: 1px solid #F5DFB0; border-radius: 10px;
        padding: 10px 12px; color: #6F4E10; }
#emptyState { background: #FAFAFD; border: 1px dashed #D6DAE8; border-radius: 12px; }

QTabWidget#mainTabs::pane { border: none; background: transparent; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab { background: #EDEFF6; color: #66708A; border: 1px solid #E3E7F0; border-bottom: 3px solid transparent;
               padding: 11px 24px; margin-right: 7px; border-top-left-radius: 10px;
               border-top-right-radius: 10px; font-weight: 700; min-width: 120px; }
QTabBar::tab:selected { background: #FFFFFF; color: #6847F5; border-bottom: 3px solid #6847F5; }
QTabBar::tab:hover:!selected { background: #F5F2FF; color: #6847F5; }

QComboBox { background: #FFFFFF; border: 1px solid #DCE2EE; border-radius: 9px;
            padding: 8px 12px; color: #34405B; min-height: 20px; }
QComboBox:hover { border-color: #A99AF8; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView { background: #FFFFFF; selection-background-color: #EAE6FF;
                              selection-color: #242044; color: #34405B; border: 1px solid #DCE2EE; }

QPushButton { background: #FFFFFF; border: 1px solid #DCE2EE; border-radius: 10px;
              padding: 10px 18px; color: #34405B; font-weight: 600; min-height: 20px; }
QPushButton:hover { background: #F6F3FF; border-color: #A99AF8; }
QPushButton:pressed { background: #EDE8FF; }
QPushButton:disabled { color: #A7ADBD; background: #F0F1F5; border-color: #E8EAF0; }
QPushButton#primary { background: #FFFFFF; color: #6847F5; border: 1px solid #6847F5; }
QPushButton#primary:hover { background: #F0EDFF; }
QPushButton#primary:disabled { color: #A7ADBD; background: #F0F1F5; border-color: #E8EAF0; }
QPushButton#analyze { background: #6847F5; color: #FFFFFF; border: 1px solid #6847F5; padding: 10px 26px; }
QPushButton#analyze:hover { background: #5537D6; border-color: #5537D6; }
QPushButton#analyze:disabled { color: #FFFFFF; background: #C9C1F5; border-color: #C9C1F5; }

QTableWidget { background: #FFFFFF; alternate-background-color: #F8F9FC; border: 1px solid #E8EBF3;
               border-radius: 11px; gridline-color: transparent; selection-background-color: #EAE6FF;
               selection-color: #1D2744; outline: 0; padding: 4px; }
QTableWidget::item { padding: 9px 10px; border-bottom: 1px solid #F0F2F7; }
QTableWidget::item:selected { background: #EAE6FF; color: #242044; }
QHeaderView::section { background: #F3F4FA; color: #53607A; border: none;
                       border-bottom: 1px solid #E5E8F1; padding: 12px 10px; font-weight: 700; }

QProgressBar { border: none; border-radius: 5px; background: #E9EAF2; max-height: 9px; min-height: 9px; }
QProgressBar::chunk { background: #6847F5; border-radius: 5px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analog Modulation Classifier")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, screen.width() - 40), min(840, screen.height() - 60))
        self.setMinimumSize(900, 600)

        self.predictor = None
        self.paths: list[Path] = []
        self.rows: list[dict] = []
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[PredictionWorker] = None
        self.last_dir = ROOT
        self._syncing = False
        self._activity_base = ""
        self._activity_tick = 0

        central = QWidget()
        central.setObjectName("root")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(24, 18, 24, 10)
        outer.setSpacing(12)
        outer.addWidget(self.build_header())

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setDrawBase(False)

        # Each tab owns ONE vertical scroll area; nothing inside it scrolls on its own.
        self.analyze_scroll, analyze_layout = self.make_page()
        self.details_scroll, details_layout = self.make_page()
        self.build_analyze_tab(analyze_layout)
        self.build_details_tab(details_layout)
        self.tabs.addTab(self.analyze_scroll, "1  ·  Analyze")
        self.tabs.addTab(self.details_scroll, "2  ·  Signal details")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.setStyleSheet(STYLE)
        self.activity_timer = QTimer(self)
        self.activity_timer.setInterval(320)
        self.activity_timer.timeout.connect(self.animate_activity)

        self.reset_details_view()
        self.load_model()

    # ------------------------------------------------------------------ layout helpers
    def make_page(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        body = QWidget()
        body.setObjectName("pageBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(2, 14, 10, 20)
        layout.setSpacing(16)
        scroll.setWidget(body)
        return scroll, layout

    @staticmethod
    def label(text: str = "", name: str = "", wrap: bool = False, rich: bool = False) -> QLabel:
        lab = QLabel(text)
        if name:
            lab.setObjectName(name)
        lab.setWordWrap(wrap)
        if rich:
            lab.setTextFormat(Qt.TextFormat.RichText)
        return lab

    @staticmethod
    def card(name: str = "card", margins=(20, 18, 20, 18), spacing: int = 12) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName(name)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)
        return frame, layout

    @staticmethod
    def set_state(widget: QWidget, state: str):
        widget.setProperty("state", state)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def set_status(self, text: str, state: str = "ready"):
        self.model_status.setText(text)
        self.set_state(self.model_status, state)

    def set_activity(self, text: str, state: str = "busy", visible: bool = True):
        self.activity.setText(text)
        self.set_state(self.activity, state)
        self.activity.setVisible(visible)

    # ------------------------------------------------------------------ header
    def build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 14, 24, 14)
        brand = QVBoxLayout()
        brand.setSpacing(2)
        brand.addWidget(self.label("Analog Modulation Classifier", "title"))
        brand.addWidget(self.label("SIGNAL ANALYSIS  /  MACHINE LEARNING", "eyebrow"))
        layout.addLayout(brand, 1)
        self.model_status = self.label("Loading model…", "status")
        layout.addWidget(self.model_status, 0, Qt.AlignmentFlag.AlignVCenter)
        return header

    # ------------------------------------------------------------------ tab 1
    def build_analyze_tab(self, page: QVBoxLayout):
        intro = QHBoxLayout()
        intro.addWidget(self.label("Analyze signals", "h1"))
        intro.addStretch(1)
        self.file_count = self.label("No signals selected", "muted")
        intro.addWidget(self.file_count)
        page.addLayout(intro)

        upload_card, uc = self.card(margins=(22, 20, 22, 20), spacing=14)
        top = QHBoxLayout()
        top.setSpacing(14)
        icon = self.label("↥", "uploadIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedSize(52, 52)
        copy = QVBoxLayout()
        copy.setSpacing(2)
        copy.addWidget(self.label("Upload signal files", "cardTitle"))
        copy.addWidget(self.label("Choose one or more .npy files. Uploading does not run the model.",
                                  "muted", wrap=True))
        top.addWidget(icon)
        top.addLayout(copy, 1)
        uc.addLayout(top)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self.upload_btn = QPushButton("＋  Upload signal(s)")
        self.upload_btn.setObjectName("primary")
        self.upload_btn.clicked.connect(self.choose_files)
        self.analyze_btn = QPushButton("Analyze signal(s)")
        self.analyze_btn.setObjectName("analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        for btn in (self.upload_btn, self.analyze_btn, self.clear_btn):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            buttons.addWidget(btn)
        buttons.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(190)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        buttons.addWidget(self.progress)
        uc.addLayout(buttons)
        self.selected_files = self.label("No files selected yet.", "muted", wrap=True)
        uc.addWidget(self.selected_files)
        page.addWidget(upload_card)

        self.hint = self.label(self.default_hint(), "hint", wrap=True)
        page.addWidget(self.hint)
        self.activity = self.label("", "activity", wrap=True)
        self.activity.setVisible(False)
        page.addWidget(self.activity)

        self.results_card, rc = self.card()
        head = QHBoxLayout()
        head.addWidget(self.label("Predictions", "h2"))
        head.addStretch(1)
        self.results_summary = self.label("Results appear here after analysis", "muted")
        head.addWidget(self.results_summary)
        rc.addLayout(head)

        self.results_empty = QFrame()
        self.results_empty.setObjectName("emptyState")
        ev = QVBoxLayout(self.results_empty)
        ev.setContentsMargins(20, 26, 20, 26)
        ev.setSpacing(4)
        e1 = self.label("No predictions yet", "emptyTitle")
        e2 = self.label("Upload one or more .npy signals, then click “Analyze signal(s)”.", "muted", wrap=True)
        e1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        e2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ev.addWidget(e1)
        ev.addWidget(e2)
        rc.addWidget(self.results_empty)

        self.table = StaticTable(0, 4)
        self.table.setHorizontalHeaderLabels(["Signal file", "Predicted class", "Model score", "Quick explanation"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(42)
        header = self.table.horizontalHeader()
        for col in range(3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        self.table.cellDoubleClicked.connect(lambda row, _col: self.open_details(row))
        self.table.setVisible(False)
        rc.addWidget(self.table)
        page.addWidget(self.results_card)
        page.addStretch(1)

    @staticmethod
    def default_hint() -> str:
        return "1  Upload signals     →     2  Click Analyze     →     3  Inspect the prediction on the Signal details tab"

    # ------------------------------------------------------------------ tab 2
    def make_plot(self, bottom_label: str, left_label: str, color: str) -> StaticPlot:
        plot = StaticPlot()
        plot.setMinimumHeight(300)
        plot.setMinimumWidth(340)
        plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot.showGrid(x=True, y=True, alpha=0.15)
        item = plot.getPlotItem()
        item.setDownsampling(auto=True, mode="peak")
        item.setClipToView(True)
        for name in ("bottom", "left"):
            axis = plot.getAxis(name)
            axis.enableAutoSIPrefix(False)      # we put units in the label ourselves
            axis.setPen(pg.mkPen("#D8DCE7"))
            axis.setTextPen(pg.mkPen("#59647C"))
            axis.setStyle(tickFont=QFont("Segoe UI", 9))
        plot.setLabel("bottom", bottom_label, **{"color": "#59647C", "font-size": "10pt"})
        plot.setLabel("left", left_label, **{"color": "#59647C", "font-size": "10pt"})
        plot._line_color = color
        return plot

    def section(self, title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        frame, lay = self.card("section", margins=(18, 15, 18, 16), spacing=8)
        lay.addWidget(self.label(title, "sectionTitle"))
        if subtitle:
            lay.addWidget(self.label(subtitle, "caption", wrap=True))
        return frame, lay

    def create_metric_card(self, title: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setObjectName("metric")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 9, 12, 10)
        lay.setSpacing(1)
        lay.addWidget(self.label(title, "metricLabel"))
        value = self.label("–", "metricValue")
        lay.addWidget(value)
        return frame, value

    def build_details_tab(self, page: QVBoxLayout):
        page.addWidget(self.label("Signal details", "h1"))

        # Empty state (shown until at least one signal is analyzed successfully)
        self.details_empty, ec = self.card(margins=(22, 40, 22, 40), spacing=6)
        self.details_empty_title = self.label("No analyzed signals yet", "emptyTitle")
        self.details_empty_text = self.label(
            "Go to the Analyze tab, upload your .npy files and click “Analyze signal(s)”. "
            "Every successfully analyzed signal will then be selectable here.", "muted", wrap=True)
        self.details_empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ec.addWidget(self.details_empty_title)
        ec.addWidget(self.details_empty_text)
        page.addWidget(self.details_empty)

        self.details_content = QWidget()
        dv = QVBoxLayout(self.details_content)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.setSpacing(16)

        # selector
        sel_card, sc = self.card(margins=(20, 16, 20, 16), spacing=8)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self.label("Signal to inspect", "sectionTitle"))
        self.detail_combo = NoWheelComboBox()
        self.detail_combo.setMinimumWidth(300)
        self.detail_combo.currentIndexChanged.connect(self.on_combo_changed)
        row.addWidget(self.detail_combo, 1)
        self.selector_count = self.label("", "muted")
        row.addWidget(self.selector_count)
        sc.addLayout(row)
        dv.addWidget(sel_card)

        # prediction summary
        sum_card, smc = self.card(margins=(22, 16, 22, 16), spacing=0)
        sum_row = QHBoxLayout()
        sum_row.setSpacing(28)
        self.sum_file = self.summary_block(sum_row, "SIGNAL FILE", "blockValue", stretch=3, wrap=True)
        self.sum_class = self.summary_block(sum_row, "PREDICTED CLASS", "blockValueBig", stretch=2)
        self.sum_score = self.summary_block(sum_row, "MODEL SCORE", "blockValueBig", stretch=2)
        self.sum_runner = self.summary_block(sum_row, "RUNNER-UP", "blockValue", stretch=2)
        smc.addLayout(sum_row)
        dv.addWidget(sum_card)

        # plots
        plot_card, pc = self.card(margins=(18, 16, 18, 16), spacing=10)
        plot_row = QHBoxLayout()
        plot_row.setSpacing(16)
        self.wave = self.make_plot("Time (ms)", "Normalized amplitude", PURPLE)
        self.spec = self.make_plot("Frequency (kHz)", "Amplitude (relative)", TEAL)
        for title, caption, plot in (
            ("Waveform", "Normalized so the peak equals 1 (display only)", self.wave),
            ("Frequency spectrum", "Hann window · DC removed · single-sided", self.spec),
        ):
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(self.label(title, "sectionTitle"))
            col.addWidget(self.label(caption, "caption"))
            col.addWidget(plot, 1)
            plot_row.addLayout(col, 1)
        pc.addLayout(plot_row)
        dv.addWidget(plot_card)

        # explanation: compact grid of short sections
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        pred_sec, pl = self.section("Prediction", "What the classifier output")
        self.exp_prediction = self.label("", wrap=True, rich=True)
        pl.addWidget(self.exp_prediction)
        pl.addStretch(1)
        mean_sec, ml = self.section("What does this mean?", "General signal theory for the predicted class")
        self.exp_meaning = self.label("", wrap=True, rich=True)
        ml.addWidget(self.exp_meaning)
        ml.addStretch(1)
        snap_sec, sl = self.section("Signal snapshot", "Measured from the loaded samples")
        snap_grid = QGridLayout()
        snap_grid.setHorizontalSpacing(10)
        snap_grid.setVerticalSpacing(10)
        self.metric_values: dict[str, QLabel] = {}
        names = ["Samples", "Sampling frequency", "Duration", "DC offset (mean)",
                 "RMS amplitude", "Peak amplitude", "Standard deviation", "Crest factor"]
        for i, name in enumerate(names):
            frame, value = self.create_metric_card(name)
            self.metric_values[name] = value
            snap_grid.addWidget(frame, i // 4, i % 4)
        for col in range(4):
            snap_grid.setColumnStretch(col, 1)
        sl.addLayout(snap_grid)
        read_sec, rl = self.section("How to read the plots")
        self.exp_plots = self.label("", wrap=True, rich=True)
        rl.addWidget(self.exp_plots)
        rl.addStretch(1)
        grid.addWidget(pred_sec, 0, 0)
        grid.addWidget(mean_sec, 0, 1)
        grid.addWidget(snap_sec, 1, 0, 1, 2)
        grid.addWidget(read_sec, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        dv.addLayout(grid)

        page.addWidget(self.details_content)
        page.addStretch(1)

    def summary_block(self, row: QHBoxLayout, title: str, value_name: str, stretch: int, wrap: bool = False) -> QLabel:
        block = QWidget()
        block.setObjectName("summaryBlock")
        lay = QVBoxLayout(block)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.label(title, "blockLabel"))
        value = self.label("–", value_name, wrap=wrap)
        value.setMinimumWidth(40)
        value.setSizePolicy(QSizePolicy.Policy.Ignored if not wrap else QSizePolicy.Policy.Preferred,
                            QSizePolicy.Policy.Preferred)
        lay.addWidget(value)
        row.addWidget(block, stretch)
        return value

    # ------------------------------------------------------------------ model / files
    def load_model(self):
        pg.setConfigOptions(antialias=True)
        if BACKEND_ERROR:
            self.set_status("Model unavailable", "error")
            QMessageBox.warning(self, "Prediction backend unavailable",
                                f"Could not import predict.py:\n\n{BACKEND_ERROR}")
            return
        try:
            self.predictor = SignalPredictor(PredictConfig(models_dir=ROOT / "trained_models"))
            model_name = getattr(getattr(self.predictor, "_artifacts", None), "best_model_name", "Saved model")
            self.set_status(f"Model ready · {model_name}", "ready")
        except Exception as exc:
            self.set_status("Model unavailable", "error")
            QMessageBox.warning(self, "Could not load model", str(exc))

    def choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose signal file(s)", str(self.last_dir), "NumPy signals (*.npy)")
        if not files:
            return
        self.last_dir = Path(files[0]).parent
        existing = {str(p.resolve()) for p in self.paths}
        for filename in files:
            path = Path(filename)
            key = str(path.resolve())
            if key not in existing:
                self.paths.append(path)
                existing.add(key)
        self.update_selected_files()
        self.analyze_btn.setEnabled(bool(self.paths) and self.predictor is not None)
        self.hint.setText("Files are selected but not analyzed yet. Click “Analyze signal(s)” when ready.")
        if self.predictor:
            self.set_status("Signals selected — ready to analyze", "ready")

    def update_selected_files(self):
        self.file_count.setText(f"{len(self.paths)} signal(s) selected" if self.paths else "No signals selected")
        if not self.paths:
            self.selected_files.setText("No files selected yet.")
        elif len(self.paths) <= 4:
            self.selected_files.setText("Selected: " + "  ·  ".join(p.name for p in self.paths))
        else:
            self.selected_files.setText("Selected: " + "  ·  ".join(p.name for p in self.paths[:4])
                                        + f"  ·  and {len(self.paths) - 4} more")

    # ------------------------------------------------------------------ analysis run
    def set_busy(self, busy: bool):
        self.upload_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.analyze_btn.setEnabled((not busy) and bool(self.paths) and self.predictor is not None)
        self.analyze_btn.setText("Analyzing…" if busy else "Analyze signal(s)")
        self.progress.setVisible(busy)

    def animate_activity(self):
        self._activity_tick = (self._activity_tick + 1) % 4
        self.activity.setText(self._activity_base + "." * self._activity_tick)

    def analyze(self):
        if self.worker_thread is not None or not self.predictor or not self.paths:
            return  # blocks duplicate clicks while a run is in progress
        self.set_busy(True)
        self.progress.setRange(0, 0)   # busy indicator until the first file finishes
        self.set_status("Analyzing signals…", "busy")
        self._activity_base = "ANALYZING  ·  Extracting features and running the model"
        self._activity_tick = 0
        self.set_activity(self._activity_base, "busy")
        self.activity_timer.start()
        self.hint.setText("Your signals are being processed. Results will appear automatically.")

        self.worker_thread = QThread(self)
        self.worker = PredictionWorker(list(self.paths), self.predictor)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.start()

    def update_progress(self, current: int, total: int):
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self._activity_base = f"ANALYZING  ·  {current} of {total} signal(s) processed"

    def on_finished(self, rows: list):
        self.activity_timer.stop()
        self.rows = rows
        valid = [i for i, r in enumerate(rows) if not r["error"] and r["view"] is not None]
        failed = len(rows) - sum(1 for r in rows if not r["error"])

        self._syncing = True
        try:
            self.populate_table(rows)
            self.detail_combo.clear()
            for idx in valid:
                self.detail_combo.addItem(f"{idx + 1}. {rows[idx]['path'].name}", idx)
        finally:
            self._syncing = False

        analyzed = len(rows) - failed
        self.results_summary.setText(f"{analyzed} of {len(rows)} analyzed")
        self.file_count.setText(f"{analyzed} of {len(rows)} analyzed")
        if failed:
            self.set_activity(f"COMPLETE  ·  {analyzed} analyzed, {failed} could not be analyzed", "warn")
            self.set_status("Analysis complete (with errors)", "ready")
        else:
            self.set_activity(f"COMPLETE  ·  {analyzed} signal(s) analyzed", "success")
            self.set_status("Analysis complete", "ready")
        self.hint.setText("Select a row (double-click to open it) or use the Signal details tab. "
                          "A model score is not a guarantee of correctness.")
        self.play_results_animation()

        if valid:
            self.select_result(valid[0], source="init")
        else:
            self.reset_details_view("None of the selected signals could be displayed. "
                                    "See the error messages in the Analyze tab.")

    def on_failed(self, message: str):
        self.activity_timer.stop()
        self.set_activity("ERROR  ·  Analysis could not be completed", "error")
        self.set_status("Analysis failed", "error")
        QMessageBox.warning(self, "Analysis failed", message)

    def cleanup_worker(self):
        worker, thread = self.worker, self.worker_thread
        self.worker = None
        self.worker_thread = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()
        self.set_busy(False)

    def play_results_animation(self):
        effect = QGraphicsOpacityEffect(self.results_card)
        self.results_card.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(450)
        animation.setStartValue(0.25)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda: self.results_card.setGraphicsEffect(None))
        self._results_animation = animation
        animation.start()

    def populate_table(self, rows: list[dict]):
        self.table.clearSelection()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [row["path"].name, row["predicted"], fmt_pct(row["confidence"]), row["why"]]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                if row["error"]:
                    item.setForeground(QColor("#B42318"))
                elif c == 1:
                    item.setForeground(QColor(PURPLE))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(r, c, item)
        has_rows = bool(rows)
        self.results_empty.setVisible(not has_rows)
        self.table.setVisible(has_rows)
        if has_rows:
            # Full-height table: the page scrolls, the table never has its own scrollbar.
            head = self.table.horizontalHeader().sizeHint().height()
            self.table.setFixedHeight(head + 42 * len(rows) + 14)

    # ------------------------------------------------------------------ selection sync
    def on_table_selection(self):
        if self._syncing:
            return
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        self.select_result(selected[0].row(), source="table")

    def on_combo_changed(self, combo_index: int):
        if self._syncing or combo_index < 0:
            return
        idx = self.detail_combo.itemData(combo_index)
        if idx is not None:
            self.select_result(int(idx), source="combo")

    def select_result(self, idx: int, source: str):
        """Single entry point: keeps table row, combo entry and details view in step."""
        if idx < 0 or idx >= len(self.rows):
            return
        pos = self.detail_combo.findData(idx)
        self._syncing = True
        try:
            if source != "table":
                self.table.selectRow(idx)
            if source != "combo" and pos >= 0:
                self.detail_combo.setCurrentIndex(pos)
        finally:
            self._syncing = False
        if pos >= 0:
            self.render_details(idx)   # failed rows have no combo entry; details stay on last valid signal

    def open_details(self, row_idx: int):
        if self.detail_combo.findData(row_idx) >= 0:
            self.select_result(row_idx, source="table")
            self.tabs.setCurrentIndex(1)

    def on_tab_changed(self, index: int):
        scroll = self.analyze_scroll if index == 0 else self.details_scroll
        scroll.verticalScrollBar().setValue(0)
        QTimer.singleShot(0, lambda: scroll.verticalScrollBar().setValue(0))

    # ------------------------------------------------------------------ details rendering
    def reset_plot(self, plot: StaticPlot, x_range, y_range):
        plot.clear()
        vb = plot.getPlotItem().getViewBox()
        vb.disableAutoRange()
        vb.setRange(xRange=x_range, yRange=y_range, padding=0, disableAutoRange=True)

    def reset_details_view(self, message: str = ""):
        """Empty state: no stale plots, summary, explanation or selector entries."""
        self._syncing = True
        try:
            self.detail_combo.clear()
        finally:
            self._syncing = False
        self.reset_plot(self.wave, (0.0, 1.0), (-1.1, 1.1))
        self.reset_plot(self.spec, (0.0, FS / 2000.0), (0.0, 1.0))
        for lab in (self.sum_file, self.sum_class, self.sum_score, self.sum_runner):
            lab.setText("–")
        for lab in (self.exp_prediction, self.exp_meaning, self.exp_plots):
            lab.setText("")
        for lab in self.metric_values.values():
            lab.setText("–")
        self.selector_count.setText("")
        self.details_empty_title.setText("No analyzed signals yet" if not message else "Nothing to display")
        if message:
            self.details_empty_text.setText(message)
        else:
            self.details_empty_text.setText(
                "Go to the Analyze tab, upload your .npy files and click “Analyze signal(s)”. "
                "Every successfully analyzed signal will then be selectable here.")
        self.details_empty.setVisible(True)
        self.details_content.setVisible(False)

    def render_details(self, idx: int):
        row = self.rows[idx]
        view = row["view"]
        try:
            # 1) build every piece first ...
            valid_positions = [self.detail_combo.itemData(i) for i in range(self.detail_combo.count())]
            position = valid_positions.index(idx) + 1 if idx in valid_positions else 1
            counter = f"{position} of {len(valid_positions)}"
            ranked = row["probabilities"]
            runner = ranked[1] if len(ranked) > 1 else None
            texts = self.build_explanation(row, view)
            # 2) ... then apply them together, so file, plots and text always match.
            self.details_empty.setVisible(False)
            self.details_content.setVisible(True)
            self.selector_count.setText(counter)
            self.sum_file.setText(row["path"].name)
            self.sum_file.setToolTip(str(row["path"]))
            self.sum_class.setText(row["predicted"])
            self.sum_score.setText(fmt_pct(row["confidence"]))
            self.sum_runner.setText(f"{runner[0]} ({runner[1] * 100:.1f}%)" if runner else "Unavailable")
            self.exp_prediction.setText(texts["prediction"])
            self.exp_meaning.setText(texts["meaning"])
            self.exp_plots.setText(texts["plots"])
            for name, text in texts["metrics"].items():
                self.metric_values[name].setText(text)
            self.draw_plots(view)
        except Exception as exc:
            self.reset_details_view(f"Could not display {row['path'].name}: {exc}")

    def draw_plots(self, v: dict):
        # Fresh item + explicit ranges every time: no state carries over between signals.
        self.reset_plot(self.wave, (float(v["time_ms"][0]), float(v["time_ms"][-1])), (-1.1, 1.1))
        self.wave.plot(v["time_ms"], v["display"], pen=pg.mkPen(PURPLE, width=1.4))
        self.reset_plot(self.spec, (0.0, v["nyquist_khz"]), (0.0, v["mag_ymax"]))
        self.spec.plot(v["freq_khz"], v["magnitude"], pen=pg.mkPen(TEAL, width=1.5),
                       fillLevel=0, brush=pg.mkBrush(22, 139, 160, 45))
        # plot() may re-enable auto-range on some versions; pin the ranges again.
        vb = self.wave.getPlotItem().getViewBox()
        vb.setRange(xRange=(float(v["time_ms"][0]), float(v["time_ms"][-1])), yRange=(-1.1, 1.1),
                    padding=0, disableAutoRange=True)
        vb = self.spec.getPlotItem().getViewBox()
        vb.setRange(xRange=(0.0, v["nyquist_khz"]), yRange=(0.0, v["mag_ymax"]),
                    padding=0, disableAutoRange=True)

    def build_explanation(self, row: dict, v: dict) -> dict:
        predicted = str(row["predicted"])
        ranked = row["probabilities"]
        runner = ranked[1] if len(ranked) > 1 else None
        esc = html.escape

        pred = (
            f'<p style="margin:0 0 6px 0;"><span style="font-size:15pt;font-weight:700;color:{INK};">{esc(predicted)}</span>'
            f'&nbsp;&nbsp;<span style="font-size:15pt;font-weight:700;color:{PURPLE};">{fmt_pct(row["confidence"])}</span>'
            f'<span style="color:{MUTED};"> model score</span></p>'
        )
        if runner:
            pred += f'<p style="margin:0 0 6px 0;">Runner-up: <b>{esc(runner[0])}</b> ({runner[1] * 100:.1f}%)</p>'
        if ranked:
            scores = " &nbsp;·&nbsp; ".join(f"{esc(k)} {s * 100:.1f}%" for k, s in ranked)
            pred += f'<p style="margin:0 0 6px 0;color:{MUTED};">All scores: {scores}</p>'
        model = f" ({esc(str(row['model_name']))})" if row.get("model_name") else ""
        pred += (
            f'<p style="margin:0 0 6px 0;color:{MUTED};">The classifier{model} works from 12 numeric features '
            f'extracted from the preprocessed signal; it does not look at these plots.</p>'
            f'<p style="margin:0;color:{MUTED};">A model score is a probability estimate, '
            f'not a guarantee that the prediction is correct.</p>'
        )

        meaning = CLASS_MEANING.get(predicted, "The model assigned this class from the extracted signal features.")
        meaning = (f'<p style="margin:0 0 6px 0;">{meaning}</p>'
                   f'<p style="margin:0;color:{MUTED};">This describes the class in general. It is not '
                   f'a statement about how the model reached its decision.</p>')

        plots = (
            '<p style="margin:0 0 6px 0;"><b style="color:#6847F5;">Waveform</b> shows how the amplitude '
            'changes over time. It is scaled so the peak equals 1, for display only.</p>'
            '<p style="margin:0 0 6px 0;"><b style="color:#168BA0;">Frequency spectrum</b> shows which '
            'frequencies are present (0 to half the sampling rate). DC is removed and a Hann window '
            'reduces leakage.</p>'
            '<p style="margin:0;">A strong peak is a carrier; clusters beside it are sidebands. '
            'AM typically has a carrier plus two sidebands, DSB-SC two sidebands with little carrier, '
            'SSB a single sideband, and FM a wider spread of peaks. Real signals can blur these patterns.</p>'
        )
        if v["all_zero"]:
            plots += '<p style="margin:6px 0 0 0;color:#A52A2A;">This signal is all zeros, so the plots are flat.</p>'

        metrics = {
            "Samples": f"{v['n']:,}",
            "Sampling frequency": f"{v['fs']:,.0f} Hz",
            "Duration": f"{v['duration_s'] * 1000:.2f} ms",
            "DC offset (mean)": fmt_num(v["mean"]),
            "RMS amplitude": fmt_num(v["rms"]),
            "Peak amplitude": fmt_num(v["peak"]),
            "Standard deviation": fmt_num(v["std"]),
            "Crest factor": "n/a" if v["crest"] is None else f"{v['crest']:.3f}",
        }
        return {"prediction": pred, "meaning": meaning, "plots": plots, "metrics": metrics}

    # ------------------------------------------------------------------ clear / close
    def clear_all(self):
        if self.worker_thread is not None:
            QMessageBox.information(self, "Analysis in progress",
                                    "Wait for the current analysis to finish before clearing.")
            return
        self.activity_timer.stop()
        self.activity.setVisible(False)
        self.paths.clear()
        self.rows.clear()
        self._syncing = True
        try:
            self.table.clearSelection()
            self.table.setRowCount(0)
        finally:
            self._syncing = False
        self.table.setVisible(False)
        self.results_empty.setVisible(True)
        self.results_summary.setText("Results appear here after analysis")
        self.reset_details_view()
        self.update_selected_files()
        self.analyze_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.hint.setText(self.default_hint())
        if self.predictor:
            self.set_status(self.model_status.text().split(" · ")[0] if False else "Model ready", "ready")
        self.analyze_scroll.verticalScrollBar().setValue(0)

    def closeEvent(self, event):
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(15000)
        super().closeEvent(event)


def main() -> int:
    pg.setConfigOptions(antialias=True, background="w", foreground="#59647C")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())