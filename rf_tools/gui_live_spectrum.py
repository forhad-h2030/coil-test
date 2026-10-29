#!/usr/bin/env python3
"""Live power vs frequency bin from the tinySA (for example on the loop). Sweeps run continuously until Stop.
    python gui_live_spectrum.py
Each sweep gives one reading per bin: the level through the RBW filter at that frequency, not integrated over the bin.
Averaging is done in linear power (mW) over the last N sweeps, then shown in the chosen unit.
The settings are read at Start, so stop and start again to change them. The tinySA serial port can only be used by
one program at a time (not together with rf_profile_server.py or gui_generator_profile.py).
Save CSV (off by default) writes every raw sweep, one row per bin, to data/spectrum_<time>.csv.
Read generator (on by default) asks the generator for its current output state, mode, frequency and power once a
second and shows them with a marker on the plot. It needs no server. If the generator cannot be reached, the tinySA
part keeps working. Until you press a button it only sends queries.
Set and RF on holds one CW frequency and power (like set_cw_tone.py, up to MAX_POWER_DBM); RF off stops it. RF goes
off first, before the new settings go in, and off again if the generator rejects a command, when Stop is pressed
or when the window closes (only if this window switched it on). Each change is logged to rf_events.jsonl.
"""
import contextlib
import csv
import json
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from tinySA_IO import TinySA

LINE_COLOR = "#2a78d6"  # categorical slot 1, validated on the surface below
GEN_COLOR = "#eb6834"  # categorical slot 2: the generator marker
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
GENERATOR_IP = "192.168.24.10"
MAX_POWER_DBM = 0.0  # cap for the Set control: the tinySA input must not see more (add an attenuator above this)
DATA_DIR = Path(__file__).resolve().parent / "data"  # everything the tools record goes here (ignored by git)
EVENTS = DATA_DIR / "rf_events.jsonl"


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


def log_event(kind, freq_hz, power_dbm):
    """One line in rf_events.jsonl, the same format as set_cw_tone.py, so one file records every RF change."""
    line = {"event_id": f"{datetime.now():%Y%m%d_%H%M%S}", "kind": kind, "t": datetime.now().isoformat(),
            "freq_mhz": None if freq_hz is None else freq_hz / 1e6, "power_dbm": power_dbm}
    DATA_DIR.mkdir(exist_ok=True)
    with open(EVENTS, "a") as f:
        f.write(json.dumps(line) + "\n")


class GeneratorThread(QThread):
    """Polls the generator's settings once a second. It only sends queries until set_cw() or rf_off() queues a
    command. Those commands run here, in this thread, so the connection to the generator is never shared."""
    polled = pyqtSignal(dict)
    failed = pyqtSignal(str)    # the connection could not be made or was lost
    rejected = pyqtSignal(str)  # the generator refused a command; RF was switched off

    def __init__(self, ip):
        super().__init__()
        self.ip = ip
        self.stop_flag = threading.Event()
        self.commands = queue.Queue()
        self.rf_from_window = False  # RF was switched on from this window, so it is switched off on the way out
        self.last = (None, None)     # frequency and power of the last Set

    def set_cw(self, freq_hz, power_dbm):
        self.commands.put(("on", freq_hz, power_dbm))

    def rf_off(self):
        self.commands.put(("off", None, None))

    @staticmethod
    def _send(sma, command):
        sma.write(command)
        err = sma.query("SYSTem:ERRor?").strip()
        if not err.startswith("0"):
            raise RuntimeError(f"{command!r} rejected: {err}")

    def _run_command(self, sma, kind, freq_hz, power_dbm):
        try:
            self._send(sma, "OUTPut:STATe OFF")  # nothing is output while the new settings go in
            if kind == "on":
                self._send(sma, "SOURce:FREQuency:MODE CW")
                self._send(sma, "SOURce:POWer:MODE CW")
                self._send(sma, f"SOURce:FREQuency:CW {freq_hz:.0f}")
                self._send(sma, f"SOURce:POWer:LEVel {power_dbm}")
                self._send(sma, "OUTPut:STATe ON")
                self.rf_from_window, self.last = True, (freq_hz, power_dbm)
                log_event("on", freq_hz, power_dbm)
            else:
                self.rf_from_window = False
                log_event("off", *self.last)
        except RuntimeError as e:
            sma.write("OUTPut:STATe OFF")
            self.rf_from_window = False
            self.rejected.emit(f"{e}; RF is off")

    def run(self):
        try:
            import pyvisa  # only needed when the generator is read
            sma = pyvisa.ResourceManager().open_resource(f"TCPIP0::{self.ip}::inst0::INSTR", timeout=3000)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        try:
            while not self.stop_flag.is_set():
                self.polled.emit({
                    "rf_on": sma.query("OUTPut:STATe?").strip() in ("1", "ON"),
                    "mode": sma.query("SOURce:FREQuency:MODE?").strip().upper(),
                    "freq_hz": float(sma.query("SOURce:FREQuency:CW?")),
                    "power_dbm": float(sma.query("SOURce:POWer:LEVel?")),
                })
                try:
                    command = self.commands.get(timeout=1.0)  # waits up to a second, so a command acts at once
                except queue.Empty:
                    continue
                self._run_command(sma, *command)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            with contextlib.suppress(Exception):
                if self.rf_from_window:
                    sma.write("OUTPut:STATe OFF")
                    log_event("off", *self.last)
            with contextlib.suppress(Exception):
                sma.close()


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
        self.gen_worker = None
        self.gen = None  # latest generator reading
        self.gen_note = ""

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
        self.save = QCheckBox("save CSV")  # off by default
        self.save.toggled.connect(self.toggle_save)
        self.csv_file = self.csv_writer = None
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.status = QLabel("stopped")
        self.peak = QLabel("")
        self.saved = QLabel("")
        self.use_gen = QCheckBox("read generator")
        self.use_gen.setChecked(True)
        self.gen_ip = QLineEdit(GENERATOR_IP)
        self.gen_ip.setFixedWidth(110)
        self.gen_label = QLabel("")
        self.set_freq = _spin(QDoubleSpinBox(), 0.1, 5300, 213.0, 4, " MHz")
        self.set_power = _spin(QDoubleSpinBox(), -110, MAX_POWER_DBM, -40.0, 1, " dBm")
        self.set_btn = QPushButton("Set and RF on")
        self.off_btn = QPushButton("RF off")
        self.gen_msg = QLabel("")
        self.set_btn.clicked.connect(self.set_generator)
        self.off_btn.clicked.connect(self.generator_off)
        self.set_controls(False)

        self.plot = pg.PlotWidget(background=SURFACE)
        self.plot.setLabel("bottom", "frequency (MHz)")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        for axis in ("bottom", "left"):
            self.plot.getAxis(axis).setPen(INK_2)
            self.plot.getAxis(axis).setTextPen(INK_2)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.curve = self.plot.plot(pen=pg.mkPen(LINE_COLOR, width=2))
        self.gen_line = pg.InfiniteLine(angle=90, pen=pg.mkPen(GEN_COLOR, width=2, style=Qt.PenStyle.DashLine),
                                        label="generator", labelOpts={"color": INK_2, "position": 0.95})
        self.gen_line.setVisible(False)
        self.plot.addItem(self.gen_line)

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
        bar.addWidget(self.save)
        bar.addWidget(self.use_gen)
        bar.addWidget(self.gen_ip)
        bar.addWidget(self.start_btn)
        bar.addWidget(self.stop_btn)
        info = QHBoxLayout()
        info.addWidget(self.status, 1)
        info.addWidget(self.saved)
        info.addWidget(self.peak)
        ctrl = QHBoxLayout()
        for w in (QLabel("set generator"), self.set_freq, self.set_power, self.set_btn, self.off_btn):
            ctrl.addWidget(w)
        ctrl.addWidget(self.gen_msg, 1)
        root = QVBoxLayout()
        root.addLayout(bar)
        root.addLayout(info)
        root.addWidget(self.gen_label)
        root.addLayout(ctrl)
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
        for w in (self.center, self.span, self.bins, self.rbw, self.use_gen, self.gen_ip):
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
        if self.save.isChecked():
            self.open_csv()
        self.status.setText("opening the tinySA...")
        self.worker.start()
        if self.use_gen.isChecked():
            self.gen_note = "generator: connecting..."
            self.gen_worker = GeneratorThread(self.gen_ip.text().strip())
            self.gen_worker.polled.connect(self.on_generator)
            self.gen_worker.failed.connect(self.on_generator_failed)
            self.gen_worker.rejected.connect(self.gen_msg.setText)
            self.gen_worker.start()
        else:
            self.gen_note = ""
        self.gen = None
        self.show_generator()

    def on_finished(self):
        self.set_running(False)
        self.close_csv()
        self.stop_generator()
        if not self.status.text().startswith("failed"):
            self.status.setText("stopped")

    def stop(self):
        if self.worker:
            self.status.setText("stopping...")
            self.worker.stop_flag.set()
            if self.gen_worker:
                self.gen_worker.stop_flag.set()

    def stop_generator(self):
        if self.gen_worker and self.gen_worker.isRunning():
            self.gen_worker.stop_flag.set()
            self.gen_worker.wait(5000)
        self.gen, self.gen_note = None, ""
        self.set_controls(False)
        self.gen_msg.setText("")
        self.show_generator()

    def on_generator(self, reading):
        self.gen = reading
        self.set_controls(True)
        self.show_generator()

    def on_generator_failed(self, msg):
        self.gen, self.gen_note = None, f"generator: not reachable ({msg})"
        self.set_controls(False)
        self.show_generator()

    def set_controls(self, enabled):
        for w in (self.set_freq, self.set_power, self.set_btn, self.off_btn):
            w.setEnabled(enabled)

    def set_generator(self):
        if self.gen_worker and self.gen_worker.isRunning():
            self.gen_msg.setText("")
            self.gen_worker.set_cw(self.set_freq.value() * 1e6, self.set_power.value())

    def generator_off(self):
        if self.gen_worker and self.gen_worker.isRunning():
            self.gen_msg.setText("")
            self.gen_worker.rf_off()

    def power_text(self, dbm):
        return f"{dbm:.1f} dBm" if self.unit.currentText() == "dBm" else f"{10 ** (dbm / 10):.3g} mW"

    def show_generator(self):
        g = self.gen
        cw_on = bool(g) and g["rf_on"] and g["mode"] == "CW"
        self.gen_line.setVisible(cw_on)
        if not g:
            self.gen_label.setText(self.gen_note)
            return
        state = "RF on" if g["rf_on"] else "RF off"
        if g["mode"] != "CW":
            self.gen_label.setText(f"generator: {state}   {g['mode']} mode (frequency and power follow its list)")
            return
        text = f"generator: {state}   CW {g['freq_hz'] / 1e6:.4f} MHz   set {self.power_text(g['power_dbm'])}"
        if cw_on:
            self.gen_line.setValue(g["freq_hz"] / 1e6)
            if self.history:  # measured level of the bin nearest the generator's frequency
                mw = np.mean(self.history, axis=0)
                i = int(np.argmin(np.abs(self.freqs_mhz * 1e6 - g["freq_hz"])))
                spacing = abs(self.freqs_mhz[1] - self.freqs_mhz[0]) * 1e6 if len(mw) > 1 else 0
                if abs(self.freqs_mhz[i] * 1e6 - g["freq_hz"]) <= spacing:
                    dbm = 10 * np.log10(mw[i])
                    text += f"   measured {self.power_text(dbm)} ({dbm - g['power_dbm']:+.1f} dB)"
        self.gen_label.setText(text)

    def toggle_save(self, on):
        if self.worker and self.worker.isRunning():  # otherwise it takes effect at the next Start
            self.open_csv() if on else self.close_csv()

    def open_csv(self):
        DATA_DIR.mkdir(exist_ok=True)
        name = DATA_DIR / f"spectrum_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.csv_file = open(name, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["time", "sweep", "freq_mhz", "dbm", "mw"])
        self.saved.setText(f"saving to data/{name.name}")

    def close_csv(self):
        if self.csv_file:
            self.csv_file.close()
            self.saved.setText(f"saved data/{Path(self.csv_file.name).name}")
        self.csv_file = self.csv_writer = None

    def on_sweep(self, rows, seconds):
        if self.csv_writer:
            now = datetime.now().isoformat(timespec="milliseconds")
            self.csv_writer.writerows([now, self.count + 1, f"{r[1] / 1e6:.6f}", f"{r[2]:.3f}", f"{r[3]:.6g}"]
                                      for r in rows)
            self.csv_file.flush()
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
        self.show_generator()
        if self.tabs.currentWidget() is self.table:  # a long table is only refreshed while it is visible
            self.table.setRowCount(len(mw))
            for r, (f, d, m) in enumerate(zip(self.freqs_mhz, dbm, mw)):
                for c, text in enumerate((f"{f:.4f}", f"{d:.2f}", f"{m:.3g}")):
                    self.table.setItem(r, c, QTableWidgetItem(text))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop_flag.set()
            self.worker.wait(10000)
        self.close_csv()
        self.stop_generator()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
