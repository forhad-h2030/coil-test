#!/usr/bin/env python3
"""Window for set-power-list-server.py: set power vs measured power per frequency bin.
    python gui_rf_profile.py
Start runs the server (generator + tinySA) in a background thread; Stop turns RF off and closes both.
Measured power of a step = the tinySA sample of that step's own bin taken closest to the middle of the step's
dwell (a sweep visits its bins one after another, so each bin's sample time is spread across the sweep).
"""
import contextlib
import importlib.util
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

SET_COLOR, MEAS_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated on the surface below
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
COLUMNS = ["step", "MHz", "set dBm", "measured dBm", "diff dB"]


def measure_steps(freq_hz, dwell_s, sweeps):
    """Measured dBm per step (None when no sample fell inside the step). Times are seconds since the trigger."""
    edges = np.concatenate([[0.0], np.cumsum(dwell_s)])
    out = []
    for f, a, b in zip(freq_hz, edges[:-1], edges[1:]):
        mid, best = (a + b) / 2, None
        for s in sweeps:
            rows = s["rows"]
            i = min(range(len(rows)), key=lambda k: abs(rows[k][1] - f))
            t = s["t0"] + (i + 0.5) / len(rows) * (s["t1"] - s["t0"])
            if a <= t <= b and (best is None or abs(t - mid) < best[0]):
                best = (abs(t - mid), rows[i][2])
        out.append(best[1] if best else None)
    return out


class _Log:
    """Stand-in for stdout: hands each printed line to a callback."""
    def __init__(self, emit):
        self.emit, self.buf = emit, ""

    def write(self, text):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.emit(line)
        return len(text)

    def flush(self):
        pass


class ServerThread(QThread):
    happened = pyqtSignal(str, object)
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, host, port, use_tinysa):
        super().__init__()
        self.host, self.port, self.use_tinysa = host, port, use_tinysa
        self.stop_flag = threading.Event()

    def run(self):
        try:
            spec = importlib.util.spec_from_file_location(
                "rf_server", Path(__file__).with_name("set-power-list-server.py"))
            server = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(server)
            with contextlib.redirect_stdout(_Log(self.log.emit)):
                server.serve(self.host, self.port, self.use_tinysa, on_event=self.happened.emit, stop=self.stop_flag)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF profile")
        self.worker = None
        self.profile = None
        self.sweeps = []

        self.host = QLineEdit(socket.gethostname())
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(11111)
        self.use_tinysa = QCheckBox("tinySA")
        self.use_tinysa.setChecked(True)
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.status = QLabel("stopped")
        self.triggered = QLabel("")

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)

        self.plot = pg.PlotWidget(background=SURFACE)
        self.plot.setLabel("bottom", "frequency (MHz)")
        self.plot.setLabel("left", "power (dBm)")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        for axis in ("bottom", "left"):
            self.plot.getAxis(axis).setPen(INK_2)
            self.plot.getAxis(axis).setTextPen(INK_2)
        self.plot.addLegend(labelTextColor=INK)
        ring = pg.mkPen(SURFACE, width=2)
        self.set_curve = self.plot.plot(pen=pg.mkPen(SET_COLOR, width=2), symbol="o", symbolSize=9,
                                        symbolBrush=SET_COLOR, symbolPen=ring, name="set")
        self.meas_curve = self.plot.plot(pen=None, symbol="d", symbolSize=11,
                                         symbolBrush=MEAS_COLOR, symbolPen=ring, name="measured")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)

        bar = QHBoxLayout()
        for w in (QLabel("host"), self.host, QLabel("port"), self.port, self.use_tinysa, self.start_btn, self.stop_btn):
            bar.addWidget(w)
        info = QHBoxLayout()
        info.addWidget(self.status, 1)
        info.addWidget(self.triggered)
        body = QHBoxLayout()
        body.addWidget(self.table, 2)
        body.addWidget(self.plot, 3)
        root = QVBoxLayout()
        root.addLayout(bar)
        root.addLayout(info)
        root.addLayout(body, 3)
        root.addWidget(self.log, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(1000, 700)

    def set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.host, self.port, self.use_tinysa):
            w.setEnabled(not running)

    def start(self):
        self.worker = ServerThread(self.host.text(), self.port.value(), self.use_tinysa.isChecked())
        self.worker.happened.connect(self.on_event)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(lambda: self.set_running(False))
        self.set_running(True)
        self.status.setText("starting...")
        self.worker.start()

    def stop(self):
        if self.worker:
            self.status.setText("stopping...")
            self.worker.stop_flag.set()

    def on_failed(self, msg):
        self.status.setText(f"failed: {msg}")
        self.log.appendPlainText(msg)

    def on_event(self, kind, data):
        if kind == "status":
            self.status.setText(data)
        elif kind == "profile":
            self.profile, self.sweeps = data, []
            self.triggered.setText("")
            self.refresh()
        elif kind == "start":
            self.triggered.setText(f"triggered {data:%H:%M:%S.%f}"[:-3])
        elif kind == "sweep":
            self.sweeps.append(data)
            self.refresh()

    def refresh(self):
        p = self.profile
        if not p:
            return
        mhz = [f / 1e6 for f in p["freq_hz"]]
        meas = measure_steps(p["freq_hz"], p["dwell_s"], self.sweeps)
        self.table.setRowCount(len(mhz))
        for r, (f, s, m) in enumerate(zip(mhz, p["power_dbm"], meas)):
            cells = [str(r + 1), f"{f:.4f}", f"{s:.1f}", "" if m is None else f"{m:.1f}",
                     "" if m is None else f"{m - s:+.1f}"]
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.set_curve.setData(mhz, p["power_dbm"])
        pts = [(x, m) for x, m in zip(mhz, meas) if m is not None]
        self.meas_curve.setData(*zip(*pts)) if pts else self.meas_curve.setData([], [])

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
