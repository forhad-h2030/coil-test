import numpy as np

LOBE = 0.05  # the points above this fraction of the peak make up the lobe that sigma is computed over


def peak_widths(x, y, kind="power", min_snr=5.0, smooth=False):
    """The largest peak or dip of y, or None if it is under min_snr noise widths above the floor. smooth=True
    averages every 3 neighbouring points first (for a dense, noisy curve such as an NMR dip). Leave it off when the
    peak is only a few points wide, as on a tinySA trace: smoothing would lower and widen it."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = max(len(y) // 10, 2)
    edge = np.r_[y[:n], y[-n:]]
    floor = np.median(edge)
    noise = 1.4826 * np.median(np.abs(edge - floor)) or 1e-12
    r = y - floor
    rs = np.convolve(r, np.ones(3) / 3, mode="same") if smooth else r
    k = int(np.argmax(np.abs(rs)))
    if abs(rs[k]) < min_snr * noise:
        return None
    sign = np.sign(rs[k])                          # a dip is turned into a peak
    rs = rs * sign
    peak = rs[k]
    lo = hi = k
    while lo > 0 and rs[lo - 1] > LOBE * peak:
        lo -= 1
    while hi < len(rs) - 1 and rs[hi + 1] > LOBE * peak:
        hi += 1
    w = rs[lo:hi + 1]
    center = np.sum(w * x[lo:hi + 1]) / np.sum(w)
    sigma = np.sqrt(np.sum(w * (x[lo:hi + 1] - center) ** 2) / np.sum(w)) if hi - lo >= 2 else None  # None: too few points
    level3 = 0.5 if kind == "power" else 2 ** -0.5

    def width(frac):
        """Outermost points of the lobe at or above the level, interpolated out to the level. None: reaches the
        edge of the data, so the width is more than the span."""
        level = frac * peak
        inside = np.nonzero(rs[lo:hi + 1] >= level)[0] + lo
        a, b = inside[0], inside[-1]
        if a == 0 or b == len(rs) - 1:
            return None
        xa = x[a] + (rs[a] - level) / (rs[a] - rs[a - 1]) * (x[a - 1] - x[a])
        xb = x[b] + (rs[b] - level) / (rs[b] - rs[b + 1]) * (x[b + 1] - x[b])
        return xa, xb

    half, bw3 = width(0.5), width(level3)
    return {"floor": floor, "peak": peak * sign, "sigma": sigma, "center": center, "snr": peak / noise,
            "fwhm_x": half, "bw3_x": bw3, "fwhm": half[1] - half[0] if half else None,
            "bw3": bw3[1] - bw3[0] if bw3 else None, "points": int(np.sum(rs >= 0.5 * peak)), "level3": level3}
