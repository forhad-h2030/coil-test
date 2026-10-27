#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_FILE = os.path.join(HERE, "demo_rf_events.csv")   # multi-event run that contains the RF events
REF_FILE = os.path.join(HERE, "NMRave_noRF.csv")      # single no-RF event, subtracted from every RF event
                                                      # (set to None to use a no-RF event inside RUN_FILE)
RF_FROM = 250                                         # events from this 0-based index on had RF ...
PULSE_LOG = None                                      # ... or: path to pulse_server_log.csv (wins over RF_FROM)
N_EVENTS = 10                                         # how many RF events to plot (0 = all)
EVERY = 1                                             # plot every K-th RF event
SHOW = False                                          # True opens each plot in a window, one after another
OUT_DIR = os.path.join(HERE, "event_plots")           # a PNG per plotted event is saved here
PDF_FILE = os.path.join(OUT_DIR, "event_plots.pdf")   # all plotted events combined, one per page (None = no PDF)
# ------------------------------------------------------------------------

import matplotlib
if "--show" not in sys.argv and not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, HERE)
from lanl_events import read_events

TOL = 5e-4  


def rf_mask_from_log(events, path):
    """Boolean mask of events that had a pulse, and the pulse UTC time per event index."""
    ids = []
    for row in csv.DictReader(open(path)):
        m = re.fullmatch(r"(\d+)-([\d.]+)", row["event"]) if row["action"] == "ON" else None
        if m:
            ids.append((float(m.group(1)), float(m.group(2)), row["utc"]))
    mask, utc = [], {}
    for n, e in enumerate(events):
        hit = next((u for r1, r2, u in ids if r1 == e["row1"] and abs(r2 - e["row2"]) < TOL), None)
        mask.append(hit is not None)
        if hit:
            utc[n] = hit
    return np.array(mask), utc


def references(rf, how):
    """Reference event index for every event: one no-RF event each (None for 'mean')."""
    no_idx = np.flatnonzero(~rf)
    if how == "mean":
        return None
    if how.isdigit():
        k = int(how)
        if rf[k]:
            raise SystemExit(f"--ref {k}: event {k} had RF; pick a no-RF event.")
        return np.full(len(rf), k)
    if how != "nearest":
        raise SystemExit("--ref must be nearest, mean or an event number")
    ref = np.empty(len(rf), dtype=int)
    for n in range(len(rf)):
        before, after = no_idx[no_idx < n], no_idx[no_idx > n]
        ref[n] = before[-1] if len(before) else after[0]   # the one just before; else the first after
    return ref


def plot_event(freq, ref, ref_label, e, n, diff, utc):
    """Two panels for one RF event: it against its no-RF reference, and the difference."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax[0].plot(freq, ref, color="tab:blue", lw=1, label=ref_label)
    ax[0].plot(freq, e["amp"], color="tab:red", lw=1, label=f"RF event {n}")
    ax[0].legend(loc="upper right")
    ax[1].plot(freq, diff, color="tab:green", lw=1)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("Frequency (MHz)")
    for x, t in zip(ax, ("RF event and its no-RF reference", "RF event minus no-RF reference")):
        x.set_title(t)
        x.set_ylabel("Amplitude (V)")
        x.grid(alpha=0.3)
    when = f"   pulse {utc[n]}" if n in utc else ""
    fig.suptitle(f"Event {n}   ID {e['row1']:.0f}-{e['row2']:.4f}{when}", fontsize=11)
    fig.tight_layout()
    return fig


def overview(freq, amp, rf, base, diff, ref_title, path):
    """All events at once: reference, RF events, differences and a heatmap."""
    fig, ax = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    ax[0].plot(freq, base[np.flatnonzero(rf)[0]], color="tab:blue", lw=1.2)
    ax[0].set_title(ref_title)
    rf_idx = np.flatnonzero(rf)
    show = rf_idx[np.linspace(0, len(rf_idx) - 1, min(150, len(rf_idx))).astype(int)]
    ax[1].plot(freq, amp[show].T, color="tab:red", alpha=0.12, lw=0.7)
    ax[1].plot(freq, amp[rf].mean(axis=0), color="darkred", lw=1.3)
    ax[1].set_title(f"With RF: {rf.sum()} events (thin) and their mean (bold)")
    ax[2].plot(freq, diff[show].T, color="tab:green", alpha=0.12, lw=0.7)
    ax[2].plot(freq, diff[rf].mean(axis=0), color="darkgreen", lw=1.3)
    ax[2].set_title("Each RF event minus its no-RF reference")
    lim = np.percentile(np.abs(diff), 99.5)
    im = ax[3].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                      extent=[freq[0], freq[-1], 0, len(amp)])
    ax[3].set_title("Every event minus its no-RF reference (rows = events in time order)")
    ax[3].set_ylabel("Event")
    fig.colorbar(im, ax=ax, label="Amplitude (V)", pad=0.02, fraction=0.03)
    for x in ax[:3]:
        x.set_ylabel("Amplitude (V)")
        x.grid(alpha=0.3)
    ax[3].set_xlabel("Frequency (MHz)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print("wrote", path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=RUN_FILE, help=f"multi-event run (default {os.path.basename(RUN_FILE)})")
    p.add_argument("--ref-file", default=REF_FILE, help="single-event no-RF file; 'none' to use an event of the run")
    p.add_argument("--pulse-log", default=PULSE_LOG, help="pulse_server_log.csv, to find the RF events")
    p.add_argument("--rf-from", type=int, default=RF_FROM, help="events from this 0-based index on had RF")
    p.add_argument("--ref", default="nearest", help="with no --ref-file: nearest (default), mean, or an event number")
    p.add_argument("--events", type=int, default=N_EVENTS, help="RF events to plot, 0 = all (default %(default)s)")
    p.add_argument("--every", type=int, default=EVERY, help="plot every K-th RF event (default %(default)s)")
    p.add_argument("--out-dir", default=OUT_DIR, help="folder for the per-event PNGs")
    p.add_argument("--pdf", default=PDF_FILE, help="combined PDF, one page per event; 'none' to skip")
    p.add_argument("--overview", help="also write an all-events overview PNG to this path")
    p.add_argument("--show", action="store_true", help="also open each plot in a window")
    a = p.parse_args()
    ref_file = None if str(a.ref_file).lower() == "none" else a.ref_file
    pdf_path = None if not a.pdf or str(a.pdf).lower() == "none" else a.pdf

    events, _ = read_events(a.csv)
    freq = events[0]["freq"]
    amp = np.array([e["amp"] for e in events])
    utc = {}
    if a.pulse_log:
        rf, utc = rf_mask_from_log(events, a.pulse_log)
    else:
        rf = np.arange(len(events)) >= a.rf_from
    if not rf.any() or (not ref_file and rf.all()):
        raise SystemExit(f"{rf.sum()} of {len(events)} events have RF: need RF events, and a no-RF event "
                         "(inside the run, or --ref-file).")

    if ref_file:
        vec = np.array([float(x) for x in open(ref_file)][4:])
        if len(vec) != amp.shape[1]:
            raise SystemExit(f"{ref_file} has {len(vec)} points, the run has {amp.shape[1]}.")
        base = np.tile(vec, (len(events), 1))
        ref_name = lambda n: f"no-RF file {os.path.basename(ref_file)}"
        ref_title = f"No RF reference: {os.path.basename(ref_file)}"
    else:
        ref = references(rf, a.ref)
        if ref is None:
            base = np.tile(amp[~rf].mean(axis=0), (len(events), 1))
            ref_name = lambda n: f"mean of {(~rf).sum()} no-RF events"
        else:
            base = amp[ref]
            ref_name = lambda n: f"no-RF event {ref[n]}"
        ref_title = "No RF reference"
    diff = amp - base
    picks = np.flatnonzero(rf)[::max(a.every, 1)]
    if a.events:
        picks = picks[:a.events]
    print(f"{len(events)} events, {rf.sum()} with RF; plotting {len(picks)}; "
          f"reference: {os.path.basename(ref_file) if ref_file else a.ref}")

    if a.overview:
        overview(freq, amp, rf, base, diff, ref_title, a.overview)
    os.makedirs(a.out_dir, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path else None
    for n in picks:
        n = int(n)
        fig = plot_event(freq, base[n], ref_name(n), events[n], n, diff[n], utc)
        fig.savefig(os.path.join(a.out_dir, f"event_{n:04d}_{events[n]['row2']:.3f}.png"), dpi=100)
        if pdf:
            pdf.savefig(fig)
        if SHOW or a.show:
            plt.show()            # blocks until you close the window, then the next event opens
        plt.close(fig)
    if pdf:
        pdf.close()
    print(f"saved {len(picks)} plots in {a.out_dir}" + (f" and {pdf_path}" if pdf_path else ""))

if __name__ == "__main__":
    main()
