#!/usr/bin/env python3
"""Live NMR view of a run: the newest event (taken with RF applied), a no-RF baseline event, and their difference.

Give the run number (the digits in the file name, e.g. QCV3598543490.csv -> 3598543490):

    python live_nmr_event.py 3598543490

It re-reads only the end of the run file every POLL_S seconds, so it keeps up while the VI is
still appending events. Close the window to stop (Ctrl+C in the terminal also works).
"""
import argparse
import glob
import os
import sys
import time

# ---------------------------------------------------------------- settings
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/spin/UVA-Lab-Software/LANL-System/rssmt/data/config"   # where the run files are written
REF_FILE = os.path.join(HERE, "NMRave_noRF.csv")   # single no-RF NMR event used as the baseline
                                                   # (None: use an event of the run itself, see REF_EVENT)
REF_EVENT = 0                                      # with REF_FILE = None: this 0-based event of the run is the baseline
POLL_S = 1.0                                       # seconds between checks of the file
STALE_S = 5.0                                      # warn in the title when no new event arrived for this long
LATEST_PNG = os.path.join(HERE, "latest_event.png")  # also saved at every update (None to skip)
# ------------------------------------------------------------------------

import matplotlib
if "--no-window" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, HERE)
from lanl_events import read_events, read_latest


def find_run(run, data_dir):
    if os.path.isfile(run):
        return run
    hits = sorted(glob.glob(os.path.join(data_dir, f"*{run}*.csv")), key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run number (digits in the file name) or the path of the run file")
    p.add_argument("--data-dir", default=DATA_DIR, help="folder searched for the run file")
    p.add_argument("--ref-file", default=REF_FILE, help="single no-RF event file; 'none' to use an event of the run")
    p.add_argument("--ref-event", type=int, default=REF_EVENT, help="baseline event of the run when no --ref-file")
    p.add_argument("--poll", type=float, default=POLL_S, help="seconds between checks (default %(default)s)")
    p.add_argument("--png", default=LATEST_PNG, help="image rewritten at every update; 'none' to skip")
    p.add_argument("--no-window", action="store_true", help="no window, just keep the PNG up to date")
    a = p.parse_args()
    png = None if str(a.png).lower() == "none" else a.png

    path = find_run(a.run, a.data_dir)
    while path is None:
        print(f"waiting for a file with {a.run!r} in {a.data_dir} ...", flush=True)
        time.sleep(max(a.poll, 1))
        path = find_run(a.run, a.data_dir)
    print("run file:", path, flush=True)

    if str(a.ref_file).lower() != "none":
        base = np.array([float(x) for x in open(a.ref_file)][4:])
        base_name = os.path.basename(a.ref_file)
    else:
        while True:
            events, _ = read_events(path) if os.path.getsize(path) else ([], None)
            if len(events) > a.ref_event:
                break
            print(f"waiting for event {a.ref_event} of the run ...", flush=True)
            time.sleep(max(a.poll, 1))
        base, base_name = events[a.ref_event]["amp"], f"event {a.ref_event} of the run"

    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    (l_ref,) = ax[0].plot([], [], color="tab:blue", lw=1, label=f"no-RF baseline: {base_name}")
    (l_ev,) = ax[0].plot([], [], color="tab:red", lw=1, label="latest NMR event (with RF)")
    (l_diff,) = ax[1].plot([], [], color="tab:green", lw=1)
    ax[0].legend(loc="upper right")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("Frequency (MHz)")
    ax[0].set_title("Latest NMR event and no-RF baseline")
    ax[1].set_title("Latest NMR event minus baseline")
    for x in ax:
        x.set_ylabel("Amplitude (V)")
        x.grid(alpha=0.3)
    if not a.no_window:
        plt.ion()
        plt.show(block=False)

    last_id, last_new, shown = None, time.monotonic(), 0
    try:
        while a.no_window or plt.fignum_exists(fig.number):
            ev = None
            try:
                ev = read_latest(path)
            except (OSError, ValueError) as e:       # file busy or mid-write: try again next time
                print("read failed, retrying:", e, flush=True)
            if ev is not None:
                if len(ev["amp"]) != len(base):
                    sys.exit(f"baseline has {len(base)} points, the run has {len(ev['amp'])}.")
                ident = (ev["row1"], ev["row2"])
                if ident != last_id:
                    last_id, last_new, shown = ident, time.monotonic(), shown + 1
                    diff = ev["amp"] - base
                    l_ref.set_data(ev["freq"], base)
                    l_ev.set_data(ev["freq"], ev["amp"])
                    l_diff.set_data(ev["freq"], diff)
                    for x in ax:
                        x.relim()
                        x.autoscale_view()
                    print(f"{time.strftime('%H:%M:%S')} event {ident[0]:.0f}-{ident[1]:.4f}  "
                          f"max |diff| {np.abs(diff).max():.4f} V", flush=True)
                    if png:
                        fig.savefig(png, dpi=100)
            age = time.monotonic() - last_new
            stale = f"   (no new event for {age:.0f} s)" if shown and age > STALE_S else ""
            if last_id:
                fig.suptitle(f"{os.path.basename(path)}   latest event {last_id[0]:.0f}-{last_id[1]:.4f}{stale}", fontsize=11)
            if a.no_window:
                time.sleep(a.poll)
            else:
                plt.pause(a.poll)
    except KeyboardInterrupt:
        pass
    print("stopped.")


if __name__ == "__main__":
    main()
