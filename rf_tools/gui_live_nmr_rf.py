#!/usr/bin/env python3
"""Live NMR and RF profile in one window.
    python gui_live_nmr_rf.py
Top: the two NMR rows of plotting/live_nmr_event.py. Row 1 is the baseline and the latest event, row 2 their
difference. The event file is a run file that the VI keeps appending to; only its end is re-read every poll.
Pick the event file and the baseline with Browse, or press Defaults. The baseline is either a single no-RF event file
(a ...AVE.csv or NMRave_noRF.csv: four header lines, then one amplitude per line) or one event of the run itself.
Bottom: gui_live_spectrum.py unchanged, the live tinySA power per frequency bin (with its generator readout and Set
controls). Watching the NMR file and the tinySA sweeps start and stop separately.
The NMR files are only read, never written.
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QDoubleSpinBox, QWidget,
)

HERE = Path(__file__).resolve().parent
PLOTTING = HERE.parent / "plotting"
sys.path.insert(0, str(PLOTTING))  # lanl_events.py (the run-file reader) lives there
from lanl_events import read_events, read_latest  # noqa: E402
from gui_live_spectrum import MainWindow as SpectrumWindow  # noqa: E402

# ---------------------------------------------------------------- defaults (the Defaults button restores them)
DATA_ROOTS = [  # the first one that exists is searched, day subfolders included (data_2026/27/QCV....csv)
    Path(r"\\PHYS-NUC6\Users\Ptgroup\Desktop\data_2026"),  # the NMR computer, seen from the lab PC
    Path("/Users/spin/UVA-Lab-Software/LANL-System/rssmt/data/config"),  # the same files on the Mac
]
REF_FILE = PLOTTING / "NMRave_noRF.csv"  # default baseline: one no-RF NMR event
REF_EVENT = 0  # when REF_FILE is missing, this 0-based event of the run is the baseline
POLL_S = 1.0  # seconds between checks of the run file
STALE_S = 5.0  # warn when no new event arrived for this long
# ------------------------------------------------------------------------------------------------------------
FIT_WAITING = "Gaussian fit of the dip: waiting for an event"
BASE_COLOR, EVENT_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated on the surface below
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def data_root():
    """The first folder of DATA_ROOTS that can be reached, or None."""
    return next((p for p in DATA_ROOTS if p.is_dir()), None)


def find_run(text, root=None):
    """The run file for a path, a run number (digits in the file name) or, when empty, the newest run file under
    the data folder, day subfolders included. The ...AVE.csv averages are never runs: a run number matches both
    files, and the average has no events."""
    text = text.strip()
    if text and Path(text).is_file():
        return Path(text)
    root = root or data_root()
    if root is None:
        return None
    pattern = f"*{text}*.csv" if text else "QCV*.csv"
    hits = [p for p in root.rglob(pattern) if not p.stem.upper().endswith("AVE")]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def find_baseline():
    """Default baseline file: REF_FILE, else the newest *noRF*.csv under the data folder, else None."""
    if REF_FILE.is_file():
        return REF_FILE
    root = data_root()
    hits = list(root.rglob("*noRF*.csv")) if root else []
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def read_baseline_file(path):
    """Amplitudes of a single no-RF event file: four header lines, then one value per line."""
    values = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    values.append(float(line))
                except ValueError:
                    raise ValueError(f"{Path(path).name}, line {n}: '{line.strip()[:30]}' is not a number (use a run "
                                     "file like QCV....csv or a baseline file: 4 header lines, then one value per "
                                     "line)") from None
    if len(values) <= 4:
        raise ValueError(f"{Path(path).name} holds only {len(values)} numbers: no baseline values after the header")
    return np.array(values[4:])


def read_baseline(path, event_no):
    """A baseline is either a run file (event number event_no of it is used) or a single-event baseline file."""
    name = Path(path).name
    try:
        events = read_events(path)[0]
    except ValueError:  # not a run file: try it as a single-event baseline file (an OSError is raised as it is)
        events = []
    if events:
        if len(events) <= event_no:
            raise ValueError(f"{name} is a run file with {len(events)} events: no event {event_no} to use "
                             "as the baseline")
        return events[event_no]["amp"], f"event {event_no} of {name}"
    return read_baseline_file(path), name


def fit_gaussian(freq, diff, min_snr=5.0):
    """Fit amp * exp(-(f - f0)^2 / (2 sigma^2)) to the dip or peak of diff (event minus baseline), in MHz and V.
    The fit is a parabola through ln|y| over the part of the curve above 20 % of the peak (so no scipy is needed
    and the noise at the edges does not pull it). Returns None if there is no clear Gaussian, else a dict with
    f0, sigma, amp, fwhm (-6 dB for an amplitude), bw3 (-3 dB: amplitude at 0.707 of the peak), snr."""
    freq, diff = np.asarray(freq, float), np.asarray(diff, float)
    edge = np.r_[diff[:len(diff) // 10], diff[-(len(diff) // 10):]]
    offset = np.median(edge)
    noise = 1.4826 * np.median(np.abs(edge - offset)) or 1e-12
    y = diff - offset
    k = int(np.argmax(np.abs(y)))
    peak = y[k]
    if abs(peak) < min_snr * noise:
        return None
    y = y / peak                                   # the main lobe is now positive and peaks at 1
    lobe = y > 0.2
    lo = k
    while lo > 0 and lobe[lo - 1]:
        lo -= 1
    hi = k
    while hi < len(y) - 1 and lobe[hi + 1]:
        hi += 1
    if hi - lo < 4:
        return None                                # too few points above 20 %
    x, ln = freq[lo:hi + 1] - freq[k], np.log(y[lo:hi + 1])
    a, b, c = np.polyfit(x, ln, 2, w=y[lo:hi + 1])  # the strong points count most
    if a >= 0:
        return None
    sigma = (-1 / (2 * a)) ** 0.5
    f0, amp = freq[k] - b / (2 * a), peak * np.exp(c - b * b / (4 * a))
    for _ in range(20):                            # polish with Gauss-Newton on the whole curve (the log fit is biased)
        g = np.exp(-(freq - f0) ** 2 / (2 * sigma ** 2))
        jac = np.stack([g, amp * g * (freq - f0) / sigma ** 2, amp * g * (freq - f0) ** 2 / sigma ** 3], axis=1)
        try:
            step = np.linalg.lstsq(jac, (diff - offset) - amp * g, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        amp, f0, sigma = amp + step[0], f0 + step[1], abs(sigma + step[2])
        if np.all(np.abs(step) < 1e-9):
            break
    return {"f0": f0, "sigma": sigma, "amp": amp,
            "fwhm": 2 * np.sqrt(2 * np.log(2)) * sigma, "bw3": 2 * np.sqrt(np.log(2)) * sigma,
            "snr": abs(peak) / noise, "offset": offset}


class NmrThread(QThread):
    """Follows a run file that is still being written and reports each new event."""
    got_event = pyqtSignal(object)  # freq, amp, base, base_name, ident (row1, row2), file
    note = pyqtSignal(str)
    warn = pyqtSignal(str)  # something is wrong but the thread keeps trying
    failed = pyqtSignal(str)

    def __init__(self, event_text, base_file, base_event, poll_s):
        super().__init__()
        self.event_text, self.base_file, self.base_event, self.poll_s = event_text, base_file, base_event, poll_s
        self.stop_flag = threading.Event()
        self.hint = ""  # why the file is not usable yet, shown while waiting

    def _wait_for(self, what, ready):
        """Poll until ready() returns something other than None. Returns None if stopped first."""
        while not self.stop_flag.is_set():
            got = ready()
            if got is not None:
                return got
            (self.warn if self.hint else self.note).emit(
                f"waiting for {what} ..." + (f"   ({self.hint})" if self.hint else ""))
            self.stop_flag.wait(max(self.poll_s, 1.0))
        return None

    def run(self):
        try:
            text = self.event_text.strip()
            self.hint = (f"nothing found for '{text}'" if text else "no QCV*.csv found") + f" under {data_root()}"
            path = self._wait_for("the run file", lambda: find_run(self.event_text))
            if path is None:
                return
            self.note.emit(f"run file: {path.name}")
            self.hint = ""
            if self.base_file:
                base, base_name = read_baseline(self.base_file, self.base_event)
            else:
                def event_amp():
                    try:
                        size = path.stat().st_size
                        events = read_events(path)[0] if size else []
                    except (OSError, ValueError) as e:  # file busy, mid-write, or not a run file
                        self.hint = f"cannot read {path.name} yet: {e}"
                        return None
                    self.hint = "" if events or not size else \
                        f"{path.name} holds no NMR events: is it a run file like QCV....csv?"
                    return events[self.base_event]["amp"] if len(events) > self.base_event else None
                base = self._wait_for(f"event {self.base_event} of the run", event_amp)
                if base is None:
                    return
                base_name = f"event {self.base_event} of the run"
            last_id = None
            while not self.stop_flag.is_set():
                try:
                    ev = read_latest(path)
                except (OSError, ValueError) as e:  # file busy or mid-write: try again next time
                    self.warn.emit(f"read failed, retrying: {e}")
                    ev = None
                if ev is None and last_id is None:
                    self.warn.emit(f"{path.name}: no complete NMR event yet (needs two events; "
                                   "is it a run file like QCV....csv?)")
                if ev is not None:
                    if len(ev["amp"]) != len(base):
                        self.failed.emit(f"the baseline has {len(base)} points, the run has {len(ev['amp'])}")
                        return
                    ident = (ev["row1"], ev["row2"])
                    if ident != last_id:
                        last_id = ident
                        self.got_event.emit({"freq": ev["freq"], "amp": ev["amp"], "base": base,
                                             "base_name": base_name, "ident": ident, "file": path.name})
                self.stop_flag.wait(self.poll_s)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


def make_plot(title):
    w = pg.PlotWidget(background=SURFACE)
    w.setTitle(title, color=INK_2, size="10pt")
    w.setLabel("left", "Amplitude (V)")
    w.showGrid(x=True, y=True, alpha=0.15)
    for axis in ("bottom", "left"):
        w.getAxis(axis).setPen(INK_2)
        w.getAxis(axis).setTextPen(INK_2)
    w.getAxis("left").enableAutoSIPrefix(False)
    return w


class NmrPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.worker, self.failed = None, False
        self.last_text, self.last_new, self.shown = "", time.monotonic(), 0

        self.event_edit = QLineEdit()
        self.event_edit.setPlaceholderText("run file, a run number, or empty for the newest run file")
        self.event_btn = QPushButton("Browse...")
        self.base_edit = QLineEdit()
        self.base_edit.setPlaceholderText("no-RF baseline file (leave empty to use an event of the run instead)")
        self.base_btn = QPushButton("Browse...")
        self.base_event = QSpinBox()
        self.base_event.setRange(0, 100000)
        self.poll = QDoubleSpinBox()
        self.poll.setRange(0.1, 60)
        self.poll.setValue(POLL_S)
        self.poll.setSuffix(" s")
        self.defaults_btn = QPushButton("Defaults")
        self.start_btn = QPushButton("Watch NMR")
        self.stop_btn = QPushButton("Stop NMR")
        self.stop_btn.setEnabled(False)
        self.status = QLabel("stopped")

        self.event_btn.clicked.connect(lambda: self.browse(self.event_edit, "Event (run) file"))
        self.base_btn.clicked.connect(lambda: self.browse(self.base_edit, "Baseline file"))
        self.defaults_btn.clicked.connect(self.defaults)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        self.top = make_plot("Latest NMR event and baseline")
        legend = self.top.addLegend(labelTextColor=INK)
        legend.anchor((0.5, 0), (0.5, 0), offset=(0, 8))  # top center: the curves leave that space free
        self.base_curve = self.top.plot(pen=pg.mkPen(BASE_COLOR, width=2), name="no-RF baseline")
        self.event_curve = self.top.plot(pen=pg.mkPen(EVENT_COLOR, width=2), name="latest NMR event (with RF)")
        self.bottom = make_plot("Latest NMR event minus baseline")
        self.bottom.setLabel("bottom", "Frequency (MHz)")
        self.bottom.setXLink(self.top)
        self.bottom.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(INK_2, width=1)))
        self.diff_curve = self.bottom.plot(pen=pg.mkPen(INK, width=2))
        self.fit_curve = self.bottom.plot(pen=pg.mkPen(EVENT_COLOR, width=2, style=Qt.PenStyle.DashLine))
        self.fit_label = QLabel(FIT_WAITING)

        def file_row(label, edit, button):
            row = QHBoxLayout()
            for w in (QLabel(label), edit, button):
                row.addWidget(w)
            row.setStretch(1, 1)
            return row

        row1 = file_row("event file", self.event_edit, self.event_btn)
        row2 = file_row("baseline file", self.base_edit, self.base_btn)
        row3 = QHBoxLayout()
        for w in (QLabel("no baseline file: use event"), self.base_event, QLabel("of the run"), QLabel("poll"),
                  self.poll):
            row3.addWidget(w)
        row3.addStretch(1)
        for w in (self.defaults_btn, self.start_btn, self.stop_btn):
            row3.addWidget(w)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(row1)
        root.addLayout(row2)
        root.addLayout(row3)
        root.addWidget(self.status)
        root.addWidget(self.fit_label)
        root.addWidget(self.top, 3)
        root.addWidget(self.bottom, 2)

        self.clock = QTimer(self)  # keeps the "no new event" warning current between events
        self.clock.timeout.connect(self.tick)
        self.clock.start(1000)
        self.defaults()

    def defaults(self):
        newest = find_run("")
        self.event_edit.setText(str(newest) if newest else "")
        base = find_baseline()
        self.base_edit.setText(str(base) if base else "")
        self.base_event.setValue(REF_EVENT)
        self.poll.setValue(POLL_S)

    def browse(self, edit, title):
        start = edit.text() or str(data_root() or HERE)
        path, _ = QFileDialog.getOpenFileName(self, title, start, "CSV files (*.csv);;All files (*)")
        if path:
            edit.setText(path)

    def set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.event_edit, self.event_btn, self.base_edit, self.base_btn, self.base_event, self.poll,
                  self.defaults_btn):
            w.setEnabled(not running)

    def start(self):
        base_file = self.base_edit.text().strip() or None
        if base_file and not Path(base_file).is_file():
            self.show_status(f"FAILED: baseline file not found: {base_file}", error=True)
            return
        self.shown = 0
        for c in (self.base_curve, self.event_curve, self.diff_curve, self.fit_curve):
            c.setData([], [])
        self.fit_label.setText(FIT_WAITING)
        self.worker = NmrThread(self.event_edit.text(), base_file, self.base_event.value(), self.poll.value())
        self.worker.got_event.connect(self.on_event)
        self.worker.note.connect(lambda msg: self.show_status(msg))
        self.worker.warn.connect(lambda msg: self.show_status(msg, error=True))
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_finished)
        self.failed = False
        self.set_running(True)
        self.show_status("starting...")
        self.worker.start()

    def on_failed(self, msg):
        self.failed = True
        self.show_status(f"FAILED: {msg}", error=True)

    def on_finished(self):
        self.set_running(False)
        if not self.failed:  # a failure message stays on screen
            self.show_status("stopped")

    def show_status(self, text, error=False):
        self.status.setStyleSheet("color: #b3261e; font-weight: bold;" if error else "")
        self.status.setText(text)

    def stop(self):
        if self.worker:
            self.show_status("stopping...")
            self.worker.stop_flag.set()

    def on_event(self, d):
        diff = d["amp"] - d["base"]
        self.base_curve.setData(d["freq"], d["base"])
        self.event_curve.setData(d["freq"], d["amp"])
        self.diff_curve.setData(d["freq"], diff)
        self.show_fit(d["freq"], diff)
        self.top.setTitle(f"Latest NMR event and baseline ({d['base_name']})", color=INK_2, size="10pt")
        self.last_text = (f"{d['file']}   latest event {d['ident'][0]:.0f}-{d['ident'][1]:.4f}   "
                          f"max |diff| {np.abs(diff).max():.4f} V")
        self.last_new, self.shown = time.monotonic(), self.shown + 1
        self.tick()

    def show_fit(self, freq, diff):
        fit = fit_gaussian(freq, diff)
        if fit is None:
            self.fit_curve.setData([], [])
            self.fit_label.setText("Gaussian fit of the dip: no clear Gaussian in event minus baseline")
            return
        gauss = np.exp(-(freq - fit["f0"]) ** 2 / (2 * fit["sigma"] ** 2))
        self.fit_curve.setData(freq, fit["offset"] + fit["amp"] * gauss)
        self.fit_label.setText(
            f"Gaussian fit of the dip:   center {fit['f0']:.4f} MHz   sigma {fit['sigma'] * 1e3:.2f} kHz   "
            f"BW at -3 dB {fit['bw3'] * 1e3:.2f} kHz   (FWHM, -6 dB: {fit['fwhm'] * 1e3:.2f} kHz)   "
            f"depth {fit['amp'] * 1e3:.2f} mV   S/N {fit['snr']:.0f}")

    def tick(self):
        if not self.last_text or not self.shown or not (self.worker and self.worker.isRunning()):
            return
        age = time.monotonic() - self.last_new
        self.show_status(self.last_text + (f"   (no new event for {age:.0f} s)" if age > STALE_S else ""), error=age > STALE_S)

    def shutdown(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_flag.set()
            self.worker.wait(5000)


class CombinedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live NMR and RF profile")
        self.nmr = NmrPanel()
        self.spectrum = SpectrumWindow()  # the existing live tinySA window, used as it is
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.nmr)
        split.addWidget(self.spectrum)
        split.setSizes([520, 520])
        self.setCentralWidget(split)
        self.resize(1200, 1050)

    def closeEvent(self, event):
        self.nmr.shutdown()
        self.spectrum.close()  # its own close stops the sweeps and the generator readout
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = CombinedWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
