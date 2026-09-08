"""
plot_xscan_comparison_260826_827_829.py
=========================================
Consistency-check comparison: mean pulse height vs. X position for
260829XY (P1->P9 scan direction), 260826XY (P9->P1, reversed), and
260827XY (P9->P1 repeat, nothing changed), all on one shared y-axis (and
shared x-axis) so the three runs are directly, visually comparable.
"""

import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

RUNS = [
    ("260829XY", "P1 -> P9"),
    ("260826XY", "P9 -> P1"),
    ("260827XY", "P9 -> P1 (repeat)"),
]


def load_results(path):
    data = defaultdict(dict)
    pad_order = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pad = row["pad"]
            if pad not in pad_order:
                pad_order.append(pad)
            x = int(row["x_mm"])
            data[pad][x] = {"mean": float(row["mean_pulse_height_mV"]), "sem": float(row["sem_mV"])}
    return data, pad_order


def main():
    loaded = []
    y_max = 0.0
    x_min, x_max = None, None
    for run, direction in RUNS:
        data, pad_order = load_results(f"hrppd_xscan_results_{run}.csv")
        xs = sorted(data[pad_order[0]].keys())
        loaded.append((run, direction, data, pad_order, xs))
        for pad in pad_order:
            vals = [data[pad][x]["mean"] + data[pad][x]["sem"] for x in xs]
            y_max = max(y_max, max(vals))
        x_min = min(xs) if x_min is None else min(x_min, min(xs))
        x_max = max(xs) if x_max is None else max(x_max, max(xs))
    y_max *= 1.08
    x_max = min(x_max, 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for i, (run, direction, data, pad_order, xs) in enumerate(loaded):
        ax = axes[i]
        ax.set_facecolor(SURFACE)
        pad_color = {pad: SLOT_COLORS[j % len(SLOT_COLORS)] for j, pad in enumerate(pad_order)}
        for pad in pad_order:
            means = [data[pad][x]["mean"] for x in xs]
            sems = [data[pad][x]["sem"] for x in xs]
            ax.errorbar(xs, means, yerr=sems, color=pad_color[pad], linewidth=2,
                        marker="o", markersize=3.5, capsize=2, elinewidth=1, label=pad, zorder=3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("X position (mm)", color=INK, fontsize=11)
        if i == 0:
            ax.set_ylabel("Mean pulse height (mV)", color=INK, fontsize=11)
        ax.set_title(f"{run}\nscan direction: {direction}", color=INK, fontsize=12, pad=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=SECONDARY, labelsize=9)
        legend = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
        for text in legend.get_texts():
            text.set_color(INK)

    fig.suptitle("Consistency check: mean pulse height vs. position, P9/P5/P1 row -- "
                  "same scan re-run in different directions (shared axes for comparison)",
                  color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = "hrppd_xscan_comparison_260826_827_829.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
