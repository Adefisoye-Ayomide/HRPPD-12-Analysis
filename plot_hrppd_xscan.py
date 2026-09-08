"""
plot_hrppd_xscan.py
====================
Plots mean pulse height vs. X position, one line per pad, from a results CSV
produced by analyze_hrppd.py.
"""

import csv, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical palette slots 1/2/3 (blue/orange/aqua), assigned in scan order
# (low X -> high X) -- fixed order per run, not tied to a specific pad name.
SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

INK        = "#0b0b0b"
SECONDARY  = "#52514e"
MUTED      = "#898781"
GRID       = "#e1e0d9"
SURFACE    = "#fcfcfb"


def load_results(path):
    data = defaultdict(dict)
    pad_order = []
    gate_pads = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pad = row["pad"]
            if pad not in pad_order:
                pad_order.append(pad)
            if row.get("gated") == "True" and pad not in gate_pads:
                gate_pads.append(pad)
            x = int(row["x_mm"])
            data[pad][x] = {
                "mean": float(row["mean_pulse_height_mV"]),
                "sem": float(row["sem_mV"]),
                "pass_rate": float(row["pass_rate"]),
                "n_selected": int(row["n_selected"]),
            }
    return data, pad_order, gate_pads


def main():
    ap = argparse.ArgumentParser(description="Plot mean pulse height vs. X position.")
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--threshold", type=float, default=-25.0,
                     help="mV threshold used by analyze_hrppd.py for this CSV, for the title only "
                          "(not re-derived from the CSV -- keep in sync with what was actually run)")
    ap.add_argument("--out-prefix", default=None,
                     help="default: derived from --results-csv")
    args = ap.parse_args()
    out_prefix = args.out_prefix or args.results_csv.replace("hrppd_xscan_results", "hrppd_xscan").rsplit(".csv", 1)[0]

    data, pad_order, gate_pads = load_results(args.results_csv)
    pad_color = {pad: SLOT_COLORS[i % len(SLOT_COLORS)] for i, pad in enumerate(pad_order)}
    xs = sorted(data[pad_order[0]].keys())

    excluded = [p for p in pad_order if p not in gate_pads]
    gate_desc = f"events selected where {' or '.join(gate_pads)} crosses {args.threshold:g} mV"
    if excluded:
        gate_desc += f"; {', '.join(excluded)} excluded from gate"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for pad in pad_order:
        means = [data[pad][x]["mean"] for x in xs]
        sems  = [data[pad][x]["sem"] for x in xs]
        ax.errorbar(xs, means, yerr=sems, color=pad_color[pad], linewidth=2,
                     marker="o", markersize=4, capsize=2, elinewidth=1,
                     label=pad, zorder=3)

    ax.set_xlabel("X position (mm)", color=INK, fontsize=11)
    ax.set_ylabel("Mean pulse height (mV)", color=INK, fontsize=11)
    ax.set_title(f"HRPPD X-scan: mean pulse height per channel\n({gate_desc})",
                  color=INK, fontsize=12, pad=12)

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)

    ax.tick_params(colors=SECONDARY, labelsize=9)
    legend = ax.legend(frameon=False, fontsize=10, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    out1 = f"{out_prefix}_pulse_height.png"
    fig.savefig(out1, facecolor=SURFACE)
    print(f"[plot] wrote {out1}")

    # Secondary diagnostic: per-channel pass rate vs position
    fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=150)
    fig2.patch.set_facecolor(SURFACE)
    ax2.set_facecolor(SURFACE)
    for pad in pad_order:
        pr = [data[pad][x]["pass_rate"] for x in xs]
        ax2.plot(xs, pr, color=pad_color[pad], linewidth=2, label=pad, zorder=3)
    ax2.set_xlabel("X position (mm)", color=INK, fontsize=11)
    ax2.set_ylabel("Fraction of selected events\nabove −5 mV", color=INK, fontsize=11)
    ax2.set_title("Per-channel threshold pass rate vs. position", color=INK, fontsize=12)
    ax2.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax2.spines[spine].set_color(MUTED)
    ax2.tick_params(colors=SECONDARY, labelsize=9)
    legend2 = ax2.legend(frameon=False, fontsize=10, loc="center right")
    for text in legend2.get_texts():
        text.set_color(INK)
    fig2.tight_layout()
    out2 = f"{out_prefix}_pass_rate.png"
    fig2.savefig(out2, facecolor=SURFACE)
    print(f"[plot] wrote {out2}")


if __name__ == "__main__":
    main()
