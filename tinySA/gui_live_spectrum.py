#!/usr/bin/env python3
"""Live power vs frequency bin from the tinySA (for example on the loop). Sweeps run continuously until Stop.
    python gui_live_spectrum.py
Each sweep gives one reading per bin: the level through the RBW filter at that frequency, not integrated over the bin.
Averaging is done in linear power (mW) over the last N sweeps, then shown in the chosen unit.
The settings are read at Start, so stop and start again to change them. The tinySA serial port can only be used by
one program at a time (not together with set-power-list-server.py or gui_generator_profile.py).
"""
import sys
import threading
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from tinySA_IO import TinySA

LINE_COLOR = "#2a78d6"  # categorical slot 1, validated on the surface below
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


class SweepThread(QThread):
    swept = pyqtSignal(object, float)  # rows [(bin, freq_hz, dbm, mw)] and the sweep time in s
    failed = pyqtSignal(str)

    def __init__(self, center_hz, span_hz, bins, rbw_hz):
        super().__init__()
        self.center_hz, self.span_hz, self.bins, self.rbw_hz = center_hz, span_hz, bins, rbw_hz
        self.stop_flag = threading.Event()

    def run(self):
        try:
            with TinySA() as sa:
                while not self.stop_flag.is_set():
                    t = time.perf_counter()
                    rows = sa.read(self.center_hz, self.span_hz, self.bins, self.rbw_hz)
                    self.swept.emit(rows, time.perf_counter() - t)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


def _spin(box, lo, hi, value, decimals, suffix):
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setSuffix(suffix)
    return box


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power vs frequency bin")
        self.worker = None
        self.history = deque(maxlen=1)
        self.count = 0

        self.center = _spin(QDoubleSpinBox(), 0.1, 5300, 213.0, 4, " MHz")
        self.span = _spin(QDoubleSpinBox(), 1, 100000, 400, 1, " kHz")
        self.bins = QSpinBox()
        self.bins.setRange(2, 2000)
        self.bins.setValue(41)
        self.rbw = _spin(QDoubleSpinBox(), 0.2, 850, 10, 1, " kHz")
        self.average = QSpinBox()
        self.average.setRange(1, 200)
        self.average.setValue(1)
        self.average.valueChanged.connect(self.set_average)
        self.unit = QComboBox()
        self.unit.addItems(["dBm", "mW"])
        self.unit.currentTextChanged.connect(self.apply_unit)
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.status = QLabel("stopped")
        self.peak = QLabel("")

        self.plot = pg.PlotWidget(background=SURFACE)
        self.plot.setLabel("bottom", "frequency (MHz)")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        for axis in ("bottom", "left"):
            self.plot.getAxis(axis).setPen(INK_2)
            self.plot.getAxis(axis).setTextPen(INK_2)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.curve = self.plot.plot(pen=pg.mkPen(LINE_COLOR, width=2))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["MHz", "dBm", "mW"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.plot, "Plot")
        self.tabs.addTab(self.table, "Table")
        self.tabs.currentChanged.connect(lambda _: self.show_trace() if self.history else None)

        bar = QHBoxLayout()
        for text, w in (("center", self.center), ("span", self.span), ("bins", self.bins), ("RBW", self.rbw),
                        ("average", self.average), ("power in", self.unit)):
            bar.addWidget(QLabel(text))
            bar.addWidget(w)
        bar.addWidget(self.start_btn)
        bar.addWidget(self.stop_btn)
        info = QHBoxLayout()
        info.addWidget(self.status, 1)
        info.addWidget(self.peak)
        root = QVBoxLayout()
        root.addLayout(bar)
        root.addLayout(info)
        root.addWidget(self.tabs, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(1000, 650)
        self.apply_unit()

    def set_average(self, n):
        self.history = deque(self.history, maxlen=n)

    def apply_unit(self, *_):
        unit = self.unit.currentText()
        self.plot.setLabel("left", f"power ({unit})")
        self.plot.setLogMode(y=unit == "mW")  # mW spans decades, so the axis is logarithmic
        if self.history:
            self.show_trace()

    def set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.center, self.span, self.bins, self.rbw):
            w.setEnabled(not running)

    def start(self):
        self.history = deque(maxlen=self.average.value())
        self.count = 0
        self.worker = SweepThread(self.center.value() * 1e6, self.span.value() * 1e3, self.bins.value(),
                                  self.rbw.value() * 1e3)
        self.worker.swept.connect(self.on_sweep)
        self.worker.failed.connect(lambda msg: self.status.setText(f"failed: {msg}"))
        self.worker.finished.connect(self.on_finished)
        self.set_running(True)
        self.status.setText("opening the tinySA...")
        self.worker.start()

    def on_finished(self):
        self.set_running(False)
        if not self.status.text().startswith("failed"):
            self.status.setText("stopped")

    def stop(self):
        if self.worker:
            self.status.setText("stopping...")
            self.worker.stop_flag.set()

    def on_sweep(self, rows, seconds):
        self.freqs_mhz = np.array([r[1] for r in rows]) / 1e6
        self.history.append(np.array([r[3] for r in rows]))
        self.count += 1
        self.show_trace()
        if self.worker and self.worker.stop_flag.is_set():
            return
        self.status.setText(f"sweep {self.count}   {seconds * 1000:.0f} ms   "
                            f"{len(self.history)}/{self.history.maxlen} averaged")

    def show_trace(self):
        mw = np.mean(self.history, axis=0)  # average in linear power, not in dB
        dbm = 10 * np.log10(mw)
        self.curve.setData(self.freqs_mhz, mw if self.unit.currentText() == "mW" else dbm)
        i = int(np.argmax(mw))
        self.peak.setText(f"peak {self.freqs_mhz[i]:.4f} MHz   {dbm[i]:.1f} dBm   {mw[i]:.3g} mW")
        if self.tabs.currentWidget() is self.table:  # a long table is only refreshed while it is visible
            self.table.setRowCount(len(mw))
            for r, (f, d, m) in enumerate(zip(self.freqs_mhz, dbm, mw)):
                for c, text in enumerate((f"{f:.4f}", f"{d:.2f}", f"{m:.3g}")):
                    self.table.setItem(r, c, QTableWidgetItem(text))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop_flag.set()
            self.worker.wait(10000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
