#!/usr/bin/env python3
"""Standalone tinySA recorder: sweep over and over, one timestamped row per sweep in events.csv.
Config fields come first, then one dBm column per bin (bin_0 ... bin_N-1).
    python record_tinysa_events.py              
    python record_tinysa_events.py --interval 1 # one sweep every second
"""
import argparse
import csv
import os
import time
from datetime import datetime

from tinySA_IO import TinySA

CENTER_HZ = 213e6
SPAN_HZ = 400e3
BINS = 500
RBW_HZ = 1e3
OUT = "events.csv"
CONFIG = ["event_id", "t_sweep", "t_sweep_end", "center_mhz", "span_hz", "bins", "rbw_khz"]


def record_event(sa, event_id, center_hz, span_hz, bins, rbw_hz, out):
    """One tinySA sweep, appended as one row to the CSV."""
    t_sweep = datetime.now()
    rows = sa.read(center_hz, span_hz, bins, rbw_hz)
    t_end = datetime.now()
    header = CONFIG + [f"bin_{i}" for i, _, _, _ in rows]
    row = [event_id, t_sweep.isoformat(), t_end.isoformat(), f"{center_hz / 1e6:.6f}",
           span_hz, bins, rbw_hz / 1e3] + [f"{d:.3f}" for _, _, d, _ in rows]
    if os.path.exists(out):
        with open(out, newline="") as f:
            if next(csv.reader(f), None) != header:  # different grid, so don't mix columns
                out = f"events_{event_id}.csv"
    new_file = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)
    print(f"-> event {event_id}: {len(rows)} bins appended to {out}")
    return out  # may differ from the requested path if that file holds another grid


def main():
    p = argparse.ArgumentParser(description="Record the tinySA event by event, timestamped.")
    p.add_argument("--port", help="serial port, e.g. COM3 (default: auto-detect)")
    p.add_argument("--center", type=float, default=CENTER_HZ, help=f"center frequency, Hz (default {CENTER_HZ:g})")
    p.add_argument("--span", type=float, default=SPAN_HZ, help=f"span, Hz (default {SPAN_HZ:g})")
    p.add_argument("--bins", type=int, default=BINS, help=f"number of bins (default {BINS})")
    p.add_argument("--rbw", type=float, default=RBW_HZ / 1e3, help=f"RBW, kHz (default {RBW_HZ / 1e3:g})")
    p.add_argument("--interval", type=float, default=0, help="seconds between sweep starts (default 0, back to back)")
    p.add_argument("--out", default=OUT, help=f"CSV path (default {OUT})")
    a = p.parse_args()

    with TinySA(a.port) as sa:
        out = a.out
        print(f"Recording to {out} (Ctrl+C to stop)...")
        try:
            while True:
                start = time.monotonic()
                out = record_event(sa, f"{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3], a.center, a.span,
                                   a.bins, a.rbw * 1e3, out)
                time.sleep(max(0, a.interval - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
