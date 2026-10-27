import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(p):
    v = [float(x) for x in open(p)]
    f0, df = v[2], v[3]
    d = v[4:]
    return [f0 + i * df for i in range(len(d))], d

f, no = read("NMRave_noRF.csv")
_, rf = read("NMRave_RF.csv")
_, df = read("NMRave_diff.csv")

fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for a, y, t, c in zip(ax, (no, rf, df), ("No RF", "With RF", "Difference (RF - no RF)"), ("tab:blue", "tab:red", "tab:green")):
    a.plot(f, y, color=c, lw=1)
    a.set_title(t)
    a.set_ylabel("Amplitude (V)")
    a.grid(alpha=0.3)
ax[-1].set_xlabel("Frequency (MHz)")
fig.tight_layout()
fig.savefig("NMRave_plot.png", dpi=130)
