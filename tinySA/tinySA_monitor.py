#!/usr/bin/env python3
"""Print the tinySA power per frequency bin while the generator plays a profile.
The tinySA is opened once, then each profile goes through three steps:
    mon = Monitor()                      # open the tinySA (before any profile)
    mon.arm(freqs_hz, duration_s)        # start the sweep thread, parked until go()
    mon.go()                             # call right after the generator is triggered: it only
                                         # stamps the start time and sets a flag, so it adds no lag
    mon.wait()                           # after RF is off: let the last sweep finish
Every sweep is printed with its time since go(), so it lines up with the generator's steps.
"""
import threading
import time
from datetime import datetime

from tinySA_IO import TinySA

RBW_HZ = 10e3      # wide enough for a fast sweep over a few bins
PAD_HZ = 100e3     # scan margin used when the profile holds a single frequency
MAX_BINS = 250     # one tinySA scan


def grid_for(freqs_hz):
    """(center, span, n) of the scan that puts a bin on every profile frequency."""
    f = sorted(set(freqs_hz))
    if not f:
        raise ValueError("empty frequency list")
    if len(f) < 2:
        f = [f[0] - PAD_HZ, f[0], f[0] + PAD_HZ]
    gap = min(b - a for a, b in zip(f, f[1:]))
    n = min(int(round((f[-1] - f[0]) / gap)) + 1, MAX_BINS)
    return (f[0] + f[-1]) / 2, f[-1] - f[0], n


class Monitor:
    def __init__(self, port=None, rbw_hz=RBW_HZ):
        self.sa = TinySA(port)
        self.rbw_hz = rbw_hz
        self._go = threading.Event()
        self._cancel = threading.Event()
        self._thread = None

    def arm(self, freqs_hz, duration_s):
        """Start the sweep thread, parked until go(). Nothing is sent to the tinySA until then."""
        center, span, n = grid_for(freqs_hz)
        self._go.clear()
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run, args=(center, span, n, duration_s), daemon=True)
        self._thread.start()

    def go(self):
        """Call right after the generator is triggered. Stamps the start time and releases the thread."""
        self.t0 = time.perf_counter()
        self.start = datetime.now()
        self._go.set()

    def cancel(self):
        """Stop an armed or running pass (profile rejected, Ctrl+C)."""
        self._cancel.set()
        self._go.set()
        self.wait()

    def wait(self):
        if self._thread:
            self._thread.join()
            self._thread = None

    def close(self):
        self.cancel()
        self.sa.close()

    def _run(self, center, span, n, duration_s):
        self._go.wait()
        if self._cancel.is_set():
            return
        print(f"tinySA: generator started {self.start:%H:%M:%S.%f}"[:-3])
        sweep = 0
        while not self._cancel.is_set() and time.perf_counter() - self.t0 < duration_s:
            t_a = time.perf_counter() - self.t0
            try:
                rows = self.sa.read(center, span, n, self.rbw_hz)
            except Exception as e:
                print(f"tinySA: sweep failed: {e}")
                return
            t_b = time.perf_counter() - self.t0
            sweep += 1
            print(f"tinySA sweep {sweep}  t={t_a:.3f}-{t_b:.3f} s")
            for _, f, d, mw in rows:
                print(f"  {f / 1e6:10.4f} MHz {d:8.2f} dBm {mw:11.4g} mW")
