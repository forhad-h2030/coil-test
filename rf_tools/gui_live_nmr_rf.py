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
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QDoubleSpinBox, QWidget,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plotting"))  # lanl_events.py (the run-file reader) lives there
from lanl_events import read_events, read_latest  # noqa: E402
from gui_live_spectrum import MainWindow as SpectrumWindow  # noqa: E402

# ---------------------------------------------------------------- defaults (the Defaults button restores them)
DATA_DIR = Path("/Users/spin/UVA-Lab-Software/LANL-System/rssmt/data/config")  # where the VI writes its run files
REF_FILE = HERE.parent / "plotting" / "NMRave_noRF.csv"  # default baseline: one no-RF NMR event
REF_EVENT = 0  # when REF_FILE is missing, this 0-based event of the run is the baseline
POLL_S = 1.0  # seconds between checks of the run file
STALE_S = 5.0  # warn when no new event arrived for this long
# ------------------------------------------------------------------------------------------------------------
BASE_COLOR, EVENT_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated on the surface below
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def find_run(text, folder=DATA_DIR):
    """The run file for a path, a run number (digits in the file name) or, when empty, the newest run file.
    The ...AVE.csv averages are never runs: a run number matches both files, and the average has no events."""
    text = text.strip()
    if text and Path(text).is_file():
        return Path(text)
    pattern = f"*{text}*.csv" if text else "QCV*.csv"
    hits = [p for p in sorted(Path(folder).glob(pattern), key=lambda p: p.stat().st_mtime)
            if not p.stem.upper().endswith("AVE")]
    return hits[-1] if hits else None


def read_baseline_file(path):
    """Amplitudes of a single no-RF event file: four header lines, then one value per line."""
    with open(path) as f:
        return np.array([float(x) for x in f if x.strip()][4:])


class NmrThread(QThread):
    """Follows a run file that is still being written and reports each new event."""
    got_event = pyqtSignal(object)  # freq, amp, base, base_name, ident (row1, row2), file
    note = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, event_text, base_file, base_event, poll_s):
        super().__init__()
        self.event_text, self.base_file, self.base_event, self.poll_s = event_text, base_file, base_event, poll_s
        self.stop_flag = threading.Event()

    def _wait_for(self, what, ready):
        """Poll until ready() returns something other than None. Returns None if stopped first."""
        while not self.stop_flag.is_set():
            got = ready()
            if got is not None:
                return got
            self.note.emit(f"waiting for {what} ...")
            self.stop_flag.wait(max(self.poll_s, 1.0))
        return None

    def run(self):
        try:
            path = self._wait_for("the run file", lambda: find_run(self.event_text))
            if path is None:
                return
            self.note.emit(f"run file: {path.name}")
            if self.base_file:
                base, base_name = read_baseline_file(self.base_file), Path(self.base_file).name
            else:
                def event_amp():
                    try:
                        events = read_events(path)[0] if path.stat().st_size else []
                    except (OSError, ValueError):  # file busy or mid-write
                        return None
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
                    self.note.emit(f"read failed, retrying: {e}")
                    ev = None
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
        self.worker = None
        self.last_text, self.last_new, self.shown = "", time.monotonic(), 0

        self.event_edit = QLineEdit()
        self.event_edit.setPlaceholderText("run file, a run number, or empty for the newest run file")
        self.event_btn = QPushButton("Browse...")
        self.base_mode = QComboBox()
        self.base_mode.addItems(["file", "event of the run"])
        self.base_edit = QLineEdit()
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

        self.event_btn.clicked.connect(self.browse_event)
        self.base_btn.clicked.connect(self.browse_base)
        self.base_mode.currentTextChanged.connect(self.apply_mode)
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

        row1 = QHBoxLayout()
        for w in (QLabel("event file"), self.event_edit, self.event_btn):
            row1.addWidget(w)
        row1.setStretch(1, 1)
        row2 = QHBoxLayout()
        for w in (QLabel("baseline"), self.base_mode, self.base_edit, self.base_btn, QLabel("event"), self.base_event,
                  QLabel("poll"), self.poll, self.defaults_btn, self.start_btn, self.stop_btn):
            row2.addWidget(w)
        row2.setStretch(2, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(row1)
        root.addLayout(row2)
        root.addWidget(self.status)
        root.addWidget(self.top, 3)
        root.addWidget(self.bottom, 2)

        self.clock = QTimer(self)  # keeps the "no new event" warning current between events
        self.clock.timeout.connect(self.tick)
        self.clock.start(1000)
        self.defaults()

    def defaults(self):
        newest = find_run("")
        self.event_edit.setText(str(newest) if newest else "")
        if REF_FILE.is_file():
            self.base_mode.setCurrentText("file")
            self.base_edit.setText(str(REF_FILE))
        else:
            self.base_mode.setCurrentText("event of the run")
            self.base_edit.setText("")
        self.base_event.setValue(REF_EVENT)
        self.poll.setValue(POLL_S)
        self.apply_mode()

    def apply_mode(self, *_):
        from_file = self.base_mode.currentText() == "file"
        self.base_edit.setEnabled(from_file)
        self.base_btn.setEnabled(from_file)
        self.base_event.setEnabled(not from_file)

    def browse(self, edit, title):
        start = edit.text() or str(DATA_DIR if DATA_DIR.is_dir() else HERE)
        path, _ = QFileDialog.getOpenFileName(self, title, start, "CSV files (*.csv);;All files (*)")
        if path:
            edit.setText(path)

    def browse_event(self):
        self.browse(self.event_edit, "Event (run) file")

    def browse_base(self):
        self.browse(self.base_edit, "Baseline file")

    def set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.event_edit, self.event_btn, self.base_mode, self.poll, self.defaults_btn):
            w.setEnabled(not running)
        if running:
            for w in (self.base_edit, self.base_btn, self.base_event):
                w.setEnabled(False)
        else:
            self.apply_mode()

    def start(self):
        from_file = self.base_mode.currentText() == "file"
        base_file = self.base_edit.text().strip() if from_file else None
        if from_file and not Path(base_file or "").is_file():
            self.status.setText("baseline file not found")
            return
        self.shown = 0
        for c in (self.base_curve, self.event_curve, self.diff_curve):
            c.setData([], [])
        self.worker = NmrThread(self.event_edit.text(), base_file, self.base_event.value(), self.poll.value())
        self.worker.got_event.connect(self.on_event)
        self.worker.note.connect(self.status.setText)
        self.worker.failed.connect(lambda msg: self.status.setText(f"failed: {msg}"))
        self.worker.finished.connect(lambda: self.set_running(False))
        self.set_running(True)
        self.status.setText("starting...")
        self.worker.start()

    def stop(self):
        if self.worker:
            self.status.setText("stopping...")
            self.worker.stop_flag.set()

    def on_event(self, d):
        diff = d["amp"] - d["base"]
        self.base_curve.setData(d["freq"], d["base"])
        self.event_curve.setData(d["freq"], d["amp"])
        self.diff_curve.setData(d["freq"], diff)
        self.top.setTitle(f"Latest NMR event and baseline ({d['base_name']})", color=INK_2, size="10pt")
        self.last_text = (f"{d['file']}   latest event {d['ident'][0]:.0f}-{d['ident'][1]:.4f}   "
                          f"max |diff| {np.abs(diff).max():.4f} V")
        self.last_new, self.shown = time.monotonic(), self.shown + 1
        self.tick()

    def tick(self):
        if not self.last_text or not self.shown or not (self.worker and self.worker.isRunning()):
            return
        age = time.monotonic() - self.last_new
        self.status.setText(self.last_text + (f"   (no new event for {age:.0f} s)" if age > STALE_S else ""))

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
