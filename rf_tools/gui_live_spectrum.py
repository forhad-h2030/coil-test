#!/usr/bin/env python3
"""Live power vs frequency bin from the tinySA (for example on the loop). Sweeps run continuously until Stop.
    python gui_live_spectrum.py
Each sweep gives one reading per bin: the level through the RBW filter at that frequency, not integrated over the bin.
Averaging is done in linear power (mW) over the last N sweeps, then shown in the chosen unit.
The settings are read at Start, so stop and start again to change them. The tinySA serial port can only be used by
one program at a time (not together with to_gen_server.py or gui_generator_profile.py).
Save CSV (off by default) writes every raw sweep, one row per bin, to data/spectrum_<time>.csv.
Read generator (on by default) asks the generator for its current output state, mode, frequency and power once a
second and shows them with a marker on the plot. It needs no server. If the generator cannot be reached, the tinySA
part keeps working. It only sends queries.
When the generator is in LIST mode (a profile from to_gen_server.py), the list is read too (frequency, power and
dwell of every step, and the index the generator reports) and shown with the measurement at each step: markers on the
plot and a Profile tab. The generator power is a setpoint, so "path loss" (dB between the generator and the tinySA
input: cables, attenuators) turns it into the level the tinySA should read: set minus loss. The list queries are
best effort: one the generator refuses is skipped and said so.
This window only reads. It never sets the generator (to hold a CW tone use set_cw_tone.py, for a profile use
to_gen_server.py).
"""
import contextlib
import csv
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

from peak_widths import peak_widths
from tinySA_IO import TinySA

LINE_COLOR = "#2a78d6"  # categorical slot 1, validated on the surface below
GEN_COLOR = "#eb6834"  # categorical slot 2: the generator marker
BW_COLOR, FWHM_COLOR, SIGMA_COLOR = "#1a8f5a", "#b8860b", "#8a4fd0"  # bars: -3 dB width, FWHM, sigma
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
GENERATOR_IP = "192.168.24.10"
LIST_REREAD_S = 30  # the list is read again at least this often, and at once when its length changes
DATA_DIR = Path(__file__).resolve().parent / "data"  # everything the tools record goes here (ignored by git)


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


def _floats(text):
    return [float(v) for v in text.replace(" ", "").split(",") if v]


class GeneratorThread(QThread):
    """Polls the generator's settings once a second. It only sends queries, never a command that changes the
    generator."""
    polled = pyqtSignal(dict)
    failed = pyqtSignal(str)  # the connection could not be made or was lost

    def __init__(self, ip):
        super().__init__()
        self.ip = ip
        self.stop_flag = threading.Event()
        self.refused = set()  # optional queries the generator refused: not asked again
        self.list_cache, self.list_read_at = None, 0.0

    def _optional(self, sma, query, parse):
        """The parsed answer to a query the generator may not support, else None. A refused query is dropped
        from then on, and the error it left in the generator's queue is cleared."""
        if query in self.refused:
            return None
        try:
            return parse(sma.query(query))
        except Exception:
            self.refused.add(query)
            with contextlib.suppress(Exception):
                for _ in range(10):
                    if sma.query("SYSTem:ERRor?").strip().startswith("0"):
                        break
            return None

    def _list_info(self, sma):
        """Frequency, power and dwell of every step of the generator's list, and the index it reports (None if the
        generator does not say). The lists are read again when their length changes or after LIST_REREAD_S."""
        points = self._optional(sma, "SOURce:LIST:FREQuency:POINts?", lambda a: int(float(a)))
        if self.list_cache is None or points != self.list_cache["points"] or time.monotonic() - self.list_read_at > LIST_REREAD_S:
            freq = self._optional(sma, "SOURce:LIST:FREQuency?", _floats)
            power = self._optional(sma, "SOURce:LIST:POWer?", _floats)
            n = len(freq or [])
            dwell = self._optional(sma, "SOURce:LIST:DWELl:LIST?", _floats)  # one value per step, if set that way
            if dwell is None or len(dwell) != n:
                one = self._optional(sma, "SOURce:LIST:DWELl?", float)  # else the single dwell of every step
                dwell = None if one is None else [one] * n
            self.list_cache = {"points": points, "freq_hz": freq, "power_dbm": power, "dwell_s": dwell}
            self.list_read_at = time.monotonic()
        index = self._optional(sma, "SOURce:LIST:INDex?", lambda a: int(float(a)))
        return dict(self.list_cache, index=index)

    def run(self):
        try:
            import pyvisa  # only needed when the generator is read
            sma = pyvisa.ResourceManager().open_resource(f"TCPIP0::{self.ip}::inst0::INSTR", timeout=3000)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        try:
            while not self.stop_flag.is_set():
                reading = {
                    "rf_on": sma.query("OUTPut:STATe?").strip() in ("1", "ON"),
                    "mode": sma.query("SOURce:FREQuency:MODE?").strip().upper(),
                    "freq_hz": float(sma.query("SOURce:FREQuency:CW?")),
                    "power_dbm": float(sma.query("SOURce:POWer:LEVel?")),
                }
                reading["list"] = self._list_info(sma) if reading["mode"] == "LIST" else None
                self.polled.emit(reading)
                self.stop_flag.wait(1.0)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            with contextlib.suppress(Exception):
                sma.close()


class WidthMarks:
    """The widths of a peak (see peak_widths.py) drawn on a plot as horizontal bars with a label at one end: the
    -3 dB width (right), the FWHM (right, only when it differs from the -3 dB width) and center +- sigma (left)."""

    def __init__(self, plot):
        self.bars, self.texts = [], []
        for color, anchor in ((BW_COLOR, (0, 0.5)), (FWHM_COLOR, (0, 0.5)), (SIGMA_COLOR, (1, 0.5))):
            self.bars.append(plot.plot(pen=pg.mkPen(color, width=3)))
            self.texts.append(pg.TextItem(color=color, anchor=anchor))
            plot.addItem(self.texts[-1])
        self.clear()

    def clear(self):
        for bar in self.bars:
            bar.setData([], [])
        for t in self.texts:
            t.setVisible(False)

    def show(self, w, data_y=lambda v: v, view_y=lambda v: v):
        """w: the dict of peak_widths. data_y maps a value (in the units of the curve) to the plotted data,
        view_y maps that to the axis position (log axes)."""
        self.clear()
        bars = []
        if w["bw3_x"]:
            bars.append((0, w["bw3_x"], w["level3"], f"-3 dB: {w['bw3'] * 1e3:.1f} kHz", 1))
        if w["fwhm_x"] and w["level3"] != 0.5:  # for a power the FWHM is the -3 dB width
            bars.append((1, w["fwhm_x"], 0.5, f"FWHM: {w['fwhm'] * 1e3:.1f} kHz", 1))
        if w["sigma"] is not None:
            bars.append((2, (w["center"] - w["sigma"], w["center"] + w["sigma"]), 0.25,
                         f"\u00b1\u03c3: {w['sigma'] * 1e3:.1f} kHz", -1))
        for i, (lo, hi), level, label, side in bars:
            y = data_y(w["floor"] + w["peak"] * level)
            self.bars[i].setData([lo, hi], [y, y])
            self.texts[i].setText(label)
            self.texts[i].setPos(hi if side > 0 else lo, view_y(y))
            self.texts[i].setVisible(True)


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
        self.loss = _spin(QDoubleSpinBox(), 0, 100, 0.0, 1, " dB")  # generator to tinySA: cables, attenuators
        self.loss.setToolTip("Attenuation between the generator output and the tinySA input. The tinySA should read "
                             "the generator power minus this.")
        self.loss.valueChanged.connect(lambda _: self.refresh_generator())
        self.res_label = QLabel("")

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
        self.marks = WidthMarks(self.plot)
        self.widths_info = QLabel("")
        ring = pg.mkPen(SURFACE, width=2)
        self.set_points = self.plot.plot(pen=None, symbol="d", symbolSize=11, symbolBrush=GEN_COLOR, symbolPen=ring)
        self.step_point = self.plot.plot(pen=None, symbol="o", symbolSize=17, symbolBrush=None,
                                         symbolPen=pg.mkPen(GEN_COLOR, width=3))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["MHz", "dBm", "mW"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.profile_table = QTableWidget(0, 7)
        self.profile_table.setHorizontalHeaderLabels(
            ["step", "MHz", "set dBm", "expected dBm", "measured dBm", "diff dB", "dwell ms"])
        self.profile_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.profile_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.profile_table.verticalHeader().setVisible(False)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.plot, "Plot")
        self.tabs.addTab(self.table, "Table")
        self.tabs.addTab(self.profile_table, "Profile")
        self.tabs.currentChanged.connect(lambda _: self.refresh_generator())

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
        info.addWidget(self.res_label)
        info.addWidget(QLabel("path loss"))
        info.addWidget(self.loss)
        info.addWidget(self.saved)
        info.addWidget(self.peak)
        root = QVBoxLayout()
        root.addLayout(bar)
        root.addLayout(info)
        root.addWidget(self.gen_label)
        root.addWidget(self.widths_info)
        root.addWidget(self.tabs, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(1000, 650)
        for box in (self.span, self.bins, self.rbw):
            box.valueChanged.connect(lambda _: self.show_resolution())
        self.show_resolution()
        self.apply_unit()

    def set_average(self, n):
        self.history = deque(self.history, maxlen=n)

    def apply_unit(self, *_):
        unit = self.unit.currentText()
        self.plot.setLabel("left", f"power ({unit})")
        self.plot.setLogMode(y=unit == "mW")  # mW spans decades, so the axis is logarithmic
        self.refresh_generator()

    def set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.center, self.span, self.bins, self.rbw, self.use_gen, self.gen_ip):
            w.setEnabled(not running)

    def start(self):
        self.history = deque(maxlen=self.average.value())
        self.count = 0
        self.marks.clear()
        self.widths_info.setText("")
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
        self.show_generator()

    def on_generator(self, reading):
        self.gen = reading
        self.show_generator()

    def on_generator_failed(self, msg):
        self.gen, self.gen_note = None, f"generator: not reachable ({msg})"
        self.show_generator()

    def power_text(self, dbm):
        return f"{dbm:.1f} dBm" if self.unit.currentText() == "dBm" else f"{10 ** (dbm / 10):.3g} mW"

    def refresh_generator(self):
        self.show_trace() if self.history else self.show_generator()  # show_trace shows the generator too

    def show_resolution(self):
        """RBW and bin spacing describe different things: the RBW is how wide each reading looks (the filter), the
        spacing is how far apart the readings are. A peak narrower than either is not resolved."""
        if self.history and len(self.freqs_mhz) > 1:  # the grid of the sweep on screen
            spacing = abs(self.freqs_mhz[1] - self.freqs_mhz[0]) * 1e3
        else:
            spacing = self.span.value() / (self.bins.value() - 1)
        rbw = self.rbw.value()
        if spacing > rbw * 1.01:
            hint = "bins wider than the RBW: a narrow signal can fall between bins"
        elif spacing < rbw / 1.01:
            hint = "bins closer than the RBW: neighbouring bins overlap"
        else:
            hint = "bins as wide as the RBW"
        self.res_label.setText(f"RBW {rbw:g} kHz   bin spacing {spacing:.2f} kHz")
        self.res_label.setToolTip(hint)

    def power_loss(self):
        return self.loss.value()

    def step_rows(self, g):
        """One row per list step: (MHz, set dBm, expected dBm at the tinySA, measured dBm or None, dwell s or None)
        and the index of the step the generator reports (or None). The measured level is the bin nearest the step."""
        lst = g["list"]
        if not lst or not lst["freq_hz"] or not lst["power_dbm"] or len(lst["freq_hz"]) != len(lst["power_dbm"]):
            return [], None
        mw = np.mean(self.history, axis=0) if self.history else None
        spacing = abs(self.freqs_mhz[1] - self.freqs_mhz[0]) if mw is not None and len(mw) > 1 else 0
        rows = []
        for k, (f, p) in enumerate(zip(lst["freq_hz"], lst["power_dbm"])):
            measured = None
            if mw is not None:
                i = int(np.argmin(np.abs(self.freqs_mhz - f / 1e6)))
                if abs(self.freqs_mhz[i] - f / 1e6) <= spacing:
                    measured = 10 * np.log10(mw[i])
            dwell = lst["dwell_s"][k] if lst["dwell_s"] and k < len(lst["dwell_s"]) else None
            rows.append((f / 1e6, p, p - self.power_loss(), measured, dwell))
        index = lst["index"] if lst["index"] is not None and 0 <= lst["index"] < len(rows) else None
        return rows, index

    def show_generator(self):
        g = self.gen
        cw_on = bool(g) and g["rf_on"] and g["mode"] == "CW"
        self.gen_line.setVisible(cw_on)
        self.show_setpoints(g)
        if not g:
            self.gen_label.setText(self.gen_note)
            return
        state = "RF on" if g["rf_on"] else "RF off"
        loss = self.power_loss()
        if g["mode"] == "LIST":
            self.gen_label.setText(f"generator: {state}   LIST mode   {self.list_text(g)}")
            return
        if g["mode"] != "CW":
            self.gen_label.setText(f"generator: {state}   {g['mode']} mode")
            return
        expect = g["power_dbm"] - loss
        text = f"generator: {state}   CW {g['freq_hz'] / 1e6:.4f} MHz   set {self.power_text(g['power_dbm'])}"
        if loss:
            text += f", tinySA should read {self.power_text(expect)}"
        if cw_on:
            self.gen_line.setValue(g["freq_hz"] / 1e6)
            if self.history:  # measured level of the bin nearest the generator's frequency
                mw = np.mean(self.history, axis=0)
                i = int(np.argmin(np.abs(self.freqs_mhz * 1e6 - g["freq_hz"])))
                spacing = abs(self.freqs_mhz[1] - self.freqs_mhz[0]) * 1e6 if len(mw) > 1 else 0
                if abs(self.freqs_mhz[i] * 1e6 - g["freq_hz"]) <= spacing:
                    dbm = 10 * np.log10(mw[i])
                    text += f"   measured {self.power_text(dbm)} ({dbm - expect:+.1f} dB)"
        self.gen_label.setText(text)

    def list_text(self, g):
        """The generator's list in one line: how many steps, the ranges, the dwell and the step it reports."""
        rows, index = self.step_rows(g)
        if not rows:
            return "its list could not be read (the generator refused the list queries)"
        f, p = [r[0] for r in rows], [r[1] for r in rows]
        dwells = [r[4] for r in rows if r[4] is not None]
        dwell = "dwell unknown" if not dwells else (f"dwell {dwells[0] * 1e3:g} ms" if len(set(dwells)) == 1 else
                                                    f"dwell {min(dwells) * 1e3:g} to {max(dwells) * 1e3:g} ms")
        text = (f"{len(rows)} steps, {min(f):.4f} to {max(f):.4f} MHz, set {min(p):.1f} to {max(p):.1f} dBm, {dwell}")
        if index is None:
            return text + "   current step: not reported by the generator"
        r = rows[index]
        now = f"   step {index + 1} of {len(rows)} (index {index}): {r[0]:.4f} MHz, set {r[1]:.1f} dBm"
        if r[4] is not None:
            now += f", dwell {r[4] * 1e3:g} ms"
        return text + now

    def show_setpoints(self, g):
        """The profile setpoints (set minus path loss, the level the tinySA should read) as markers on the plot,
        and the table of steps with the measurement beside each."""
        rows, index = self.step_rows(g) if g and g.get("list") else ([], None)
        in_mw = self.unit.currentText() == "mW"
        conv = (lambda d: 10 ** (d / 10)) if in_mw else (lambda d: d)
        lo, hi = (self.freqs_mhz[0], self.freqs_mhz[-1]) if self.history else (
            self.center.value() - self.span.value() / 2e3, self.center.value() + self.span.value() / 2e3)
        seen = [(r[0], conv(r[2])) for r in rows if lo <= r[0] <= hi]  # outside the span they would stretch the plot
        self.set_points.setData(*zip(*seen)) if seen else self.set_points.setData([], [])
        now = [(rows[index][0], conv(rows[index][2]))] if index is not None and lo <= rows[index][0] <= hi else []
        self.step_point.setData(*zip(*now)) if now else self.step_point.setData([], [])
        if self.tabs.currentWidget() is not self.profile_table:  # a long table is only refreshed while it is visible
            return
        self.profile_table.setRowCount(len(rows))
        fmt = lambda v, f: "" if v is None else f.format(v)
        for r, (f, p, expect, measured, dwell) in enumerate(rows):
            diff = None if measured is None else measured - expect
            cells = [str(r + 1) + (" <" if r == index else ""), f"{f:.4f}", f"{p:.1f}", f"{expect:.1f}",
                     fmt(measured, "{:.1f}"), fmt(diff, "{:+.1f}"), fmt(None if dwell is None else dwell * 1e3, "{:g}")]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if r == index:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.profile_table.setItem(r, c, item)

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
        self.show_resolution()
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
        self.show_widths(mw)
        self.show_generator()
        if self.tabs.currentWidget() is self.table:  # a long table is only refreshed while it is visible
            self.table.setRowCount(len(mw))
            for r, (f, d, m) in enumerate(zip(self.freqs_mhz, dbm, mw)):
                for c, text in enumerate((f"{f:.4f}", f"{d:.2f}", f"{m:.3g}")):
                    self.table.setItem(r, c, QTableWidgetItem(text))

    def show_widths(self, mw):
        """Sigma, FWHM and the -3 dB width of the peak, measured on the linear power (no fit, no assumed shape)."""
        w = peak_widths(self.freqs_mhz, mw, "power", min_snr=10)  # at least 10 noise widths above the floor
        if w is None:
            self.marks.clear()
            self.widths_info.setText("Peak widths: no clear peak")
            return
        in_mw = self.unit.currentText() == "mW"
        self.marks.show(w, (lambda v: v) if in_mw else (lambda v: 10 * np.log10(np.maximum(v, 1e-30))),
                        np.log10 if in_mw else (lambda v: v))
        spacing = abs(self.freqs_mhz[1] - self.freqs_mhz[0])
        kHz = lambda v: "wider than the span" if v is None else f"{v * 1e3:.2f} kHz"
        note = ""
        if w["points"] < 3:
            note = f"   (only {w['points']} bin above half height: limited by the {spacing * 1e3:.0f} kHz bins, " \
                   "use more bins or a smaller span)"
        self.widths_info.setText(
            f"Peak widths:   center {w['center']:.4f} MHz   sigma {kHz(w['sigma']) if w['sigma'] is not None else 'n/a'}"
            f"   FWHM = BW at -3 dB {kHz(w['fwhm'])}   peak {10 * np.log10(w['peak'] + w['floor']):.1f} dBm{note}")

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
