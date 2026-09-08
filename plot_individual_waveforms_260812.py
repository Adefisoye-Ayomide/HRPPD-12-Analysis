"""
plot_individual_waveforms_260812.py
=====================================
Individual (non-overlaid) example waveforms at each pad's peak position for
the 260812 run -- one single trace per panel, laid out in a grid (rows =
example event, columns = pad/channel) so individual pulse shapes can be
compared side by side, rather than blurred together in an overlay. All
panels share one y-axis (set from the deepest example dip) for direct
amplitude comparison.

Usage:
    python3 plot_individual_waveforms_260812.py
"""

import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_hrppd import detect_rail

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
RUN = "260812"
OUT_DIR = "/home/matthew/HRPPD_characterization/overlay_plots"

CAPTURES = {
    "P14": ("AA0812X17", 2, 17),
    "P10": ("AA0812X23", 3, 40),  # mislabelled prefix; true position is 40mm
    "P6":  ("AA0812X67", 4, 67),
}

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)
DISPLAY_T_RANGE = (20.0, 90.0)
THRESHOLD_MV = -25.0
N_EXAMPLES = 5


def event_ph(t, v):
    b_lo, b_hi = BASELINE_T_RANGE
    s_lo, s_hi = SIGNAL_T_RANGE
    b = np.median(v[(t >= b_lo) & (t <= b_hi)])
    sig = (t >= s_lo) & (t <= s_hi)
    return (v[sig] - b).min(), b  # negative-going mV, baseline


def pick_example_files(cache_dir, prefix, ch, n_examples, mode="spread"):
    files = sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_ch{ch}_*.npz")))
    phs = np.empty(len(files))
    for i, f in enumerate(files):
        with np.load(f) as z:
            t, v = z["t"], z["v"]
        phs[i], _ = event_ph(t, v)
    passing = np.where(phs <= THRESHOLD_MV)[0]  # negative-going: <= -25mV is a real pulse
    if len(passing) < n_examples:
        chosen = passing
    elif mode == "top":
        # deepest (most negative -> largest amplitude) events among those passing
        order = passing[np.argsort(phs[passing])]  # most negative first
        chosen = order[:n_examples]
    else:
        idx = np.linspace(0, len(passing) - 1, n_examples).astype(int)
        chosen = passing[idx]
    return [files[i] for i in chosen]


def main():
    cache_dir = os.path.join(CACHE_ROOT, RUN)
    d_lo, d_hi = DISPLAY_T_RANGE

    # Pass 1: pick examples per pad and load their traces.
    # P14 gets its deepest (highest-amplitude) events specifically, to inspect
    # why it sometimes reads ~2x the amplitude of P10/P6; the other pads keep
    # a representative spread for comparison.
    SELECT_MODE = {"P14": "top", "P10": "spread", "P6": "spread"}
    pad_examples = {}
    for pad, (prefix, ch, x_disp) in CAPTURES.items():
        mode = SELECT_MODE.get(pad, "spread")
        chosen_files = pick_example_files(cache_dir, prefix, ch, N_EXAMPLES, mode=mode)
        traces = []
        for f in chosen_files:
            with np.load(f) as z:
                t, v = z["t"], z["v"]
            b = np.median(v[(t >= BASELINE_T_RANGE[0]) & (t <= BASELINE_T_RANGE[1])])
            m = (t >= d_lo) & (t <= d_hi)
            tm, vm = t[m], (v - b)[m]
            traces.append((tm, vm))
        pad_examples[pad] = traces
        print(f"[individual] {pad} (ch{ch}, mode={mode}): picked {len(traces)} example events")

    y_lo = -500.0
    y_hi = 20.0

    pads = list(CAPTURES.keys())
    fig, axes = plt.subplots(N_EXAMPLES, len(pads), figsize=(5.6 * len(pads), 2.6 * N_EXAMPLES), dpi=150,
                              squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for col, pad in enumerate(pads):
        prefix, ch, x_disp = CAPTURES[pad]
        color = SLOT_COLORS[col % len(SLOT_COLORS)]
        traces = pad_examples[pad]
        for row in range(N_EXAMPLES):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            if row < len(traces):
                tm, vm = traces[row]
                ax.plot(tm, vm, color=color, linewidth=1.1, zorder=3)
                ax.axvspan(*SIGNAL_T_RANGE, color=color, alpha=0.08, zorder=1)
                ax.axhline(THRESHOLD_MV, color=MUTED, linewidth=0.9, linestyle="--", zorder=2)
                ax.text(0.03, 0.05, f"min = {vm.min():.1f} mV", transform=ax.transAxes,
                        color=INK, fontsize=8, ha="left", va="bottom")
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(d_lo, d_hi)
            ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(MUTED)
            ax.tick_params(colors=SECONDARY, labelsize=8)
            if row == 0:
                tag = " -- highest-amplitude examples" if SELECT_MODE.get(pad) == "top" else ""
                ax.set_title(f"{pad} (ch{ch}), X={x_disp} mm{tag}", color=INK, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"event {row+1}\nVoltage (mV)", color=INK, fontsize=8.5)
            if row == N_EXAMPLES - 1:
                ax.set_xlabel("Time (ns)", color=INK, fontsize=9)

    fig.suptitle(f"{RUN}: individual example waveforms at each pad's peak (no overlay)",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "waveforms_260812_peak_individual.png")
    fig.savefig(out, facecolor=SURFACE)
    print(f"[individual] wrote {out}")


if __name__ == "__main__":
    main()
