"""Read a multi-event LANL CSV (many event blocks appended in one file).

Each block: some text lines, then row 1 (5-digit .0000 line), row 2, centre MHz, step MHz,
number of points, some settings, then the amplitude values. The values are the `npts`
lines just before the next block's text lines (or the end of the file).
"""
import re

import numpy as np

MARK = re.compile(r"\s*\d{5}\.0000")


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_events(path):
    """Return (events, lines). Each event is a dict with row1, row2, freq (MHz array), amp (V array), data_at."""
    lines = open(path).read().splitlines()
    starts = [i for i, l in enumerate(lines) if MARK.fullmatch(l)]
    events = []
    for k, i in enumerate(starts):
        limit = starts[k + 1] if k + 1 < len(starts) else len(lines)
        end = next((j for j in range(i + 5, limit) if lines[j].strip() and not _is_number(lines[j])), None)
        if end is None:                   # last block: the data run to the end of the file
            end = limit
            while end > i and not lines[end - 1].strip():
                end -= 1
        center, step, npts = float(lines[i + 2]), float(lines[i + 3]), int(float(lines[i + 4]))
        at = end - npts
        events.append({
            "row1": float(lines[i]), "row2": float(lines[i + 1]),
            "freq": center + (np.arange(npts) - npts // 2) * step,
            "amp": np.array([float(x) for x in lines[at:end]]),
            "data_at": at,
        })
    return events, lines


def read_latest(path, tail_bytes=262144):
    """Newest complete event in a file that is still being written, reading only its end.

    The last block may be half written. The number of settings lines is learned from the block
    before it, so the last block counts as complete once all of its values have arrived.
    Returns an event dict like read_events, or None if no complete event is in the tail yet.
    """
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - tail_bytes))
        text = f.read().decode(errors="replace")
    lines = text.splitlines()
    if size > tail_bytes:
        lines = lines[1:]                          # the first line may be cut in half
    starts = [i for i, l in enumerate(lines) if MARK.fullmatch(l)]
    if len(starts) < 2:
        return None

    def build(i, at, npts, center, step):
        return {"row1": float(lines[i]), "row2": float(lines[i + 1]),
                "freq": center + (np.arange(npts) - npts // 2) * step,
                "amp": np.array([float(x) for x in lines[at:at + npts]]), "data_at": at}

    i_prev, i_last = starts[-2], starts[-1]
    end = next((j for j in range(i_prev + 5, i_last) if lines[j].strip() and not _is_number(lines[j])), None)
    if end is None:
        return None
    center, step, npts = float(lines[i_prev + 2]), float(lines[i_prev + 3]), int(float(lines[i_prev + 4]))
    settings = (end - npts) - (i_prev + 5)

    try:                                           # is the last block complete already?
        center_l, step_l, npts_l = float(lines[i_last + 2]), float(lines[i_last + 3]), int(float(lines[i_last + 4]))
        at = i_last + 5 + settings
        tail = [l.strip() for l in lines[at:at + npts_l]]
        if len(tail) == npts_l and all(_is_number(x) for x in tail):
            return build(i_last, at, npts_l, center_l, step_l)
    except (ValueError, IndexError):
        pass
    return build(i_prev, end - npts, npts, center, step)
