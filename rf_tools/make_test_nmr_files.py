#!/usr/bin/env python3
"""Write fake NMR files to test gui_live_nmr_rf.py without the NMR computer. Real data is never touched.
    python make_test_nmr_files.py                 # a no-RF baseline file and a finished run of 20 events
    python make_test_nmr_files.py --live          # the same, then one more event every second, like the VI (Ctrl+C stops)
The files go to data/test_nmr/ next to this script (git ignores data/). In the window, press Browse for
'event file' and 'baseline file' and pick the two paths printed at the end.
The fake event is the baseline plus a dip that steps through 212.8, 212.9, 213.0, 213.1 and 213.2 MHz, one frequency
per event, so a new event is easy to see.
"""
import argparse
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "data" / "test_nmr"
BASELINE = OUT_DIR / "test_noRF_baseline.csv"
RUN = OUT_DIR / "QCV9999999991.csv"
NPTS, CENTER, STEP = 401, 213.0, 0.002
TEXT = ["Fake test run", "NMRave.csv", "Unused", "NMRave.csv", "Unused", "Tune2.csv", "", "", "", ""]  # 10 lines per block
ROW1 = "99999.0000"  # five digits and .0000, which is how the reader finds an event
rng = np.random.default_rng(1)
FREQ = CENTER + (np.arange(NPTS) - NPTS // 2) * STEP


def baseline_curve():
    """A smooth bowl, about -0.0042 V at the edges and -0.0083 V in the middle, like a real no-RF event."""
    return -0.0083 + 0.0041 * ((FREQ - CENTER) / 0.4) ** 2


def event_curve(k):
    tone = 212.8 + 0.1 * (k % 5)
    return baseline_curve() - 0.0007 - 0.003 * np.exp(-((FREQ - tone) / 0.01) ** 2) + rng.normal(0, 0.0002, NPTS)


def event_lines(k):
    head = TEXT + [ROW1, f"{1000.0 + k:10.4f}", f"{CENTER:10.4f}", f"{STEP:10.4f}", f"{NPTS:10.4f}"]
    settings = [f"{0.0:10.4f}"] * 20
    return head + settings, [f"{v:10.4f}" for v in event_curve(k)]


def write_baseline():
    """Four header lines (marker, row 2, start frequency, step), then one value per line."""
    lines = [ROW1, f"{2000.0:10.4f}", f"{FREQ[0]:10.4f}", f"{STEP:10.4f}"] + [f"{v:10.4f}" for v in baseline_curve()]
    BASELINE.write_text("\n".join(lines) + "\n")


def append_event(f, k, slow=False):
    head, values = event_lines(k)
    f.write("\n".join(head) + "\n")
    if slow:  # the VI writes an event in pieces: show the reader a half-written event for a moment
        half = len(values) // 2
        f.write("\n".join(values[:half]) + "\n")
        f.flush()
        time.sleep(0.2)
        values = values[half:]
    f.write("\n".join(values) + "\n")
    f.flush()


def main():
    p = argparse.ArgumentParser(description="Write fake NMR files for testing gui_live_nmr_rf.py.")
    p.add_argument("--live", action="store_true", help="after the first events, keep appending one event per interval")
    p.add_argument("--events", type=int, default=20, help="events written at the start (default 20)")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between live events (default 1)")
    a = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_baseline()
    with open(RUN, "w") as f:
        for k in range(a.events):
            append_event(f, k)
    print(f"event file   : {RUN}   ({a.events} events)")
    print(f"baseline file: {BASELINE}")
    if not a.live:
        return
    print(f"appending one event every {a.interval:g} s, Ctrl+C to stop ...", flush=True)
    k = a.events
    try:
        with open(RUN, "a") as f:
            while True:
                time.sleep(a.interval)
                append_event(f, k, slow=True)
                print(f"   event {k} written", flush=True)
                k += 1
    except KeyboardInterrupt:
        print("stopped.")


if __name__ == "__main__":
    main()
