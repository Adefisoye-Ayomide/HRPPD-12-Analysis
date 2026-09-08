"""
analyze_260814_boundary.py
============================
Dedicated single-position capture at the P14/P10 boundary (X=28mm, where
P14's falling edge crosses P10's rising edge in the 260806XY/260812 data) --
not a peak position for either pad. Plots the per-event pulse-height
distribution for all three pad channels at this one shared position, same
conventions as analyze_260812.py (all events, no cuts, 10mV bins, clip
detection, tight-headroom y-limit).

Usage:
    python analyze_260814_boundary.py
"""

import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_hrppd import detect_rail

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
RUN = "260814"
PREFIX = "AA0814X28"
X_DISP = 28

# pad label -> scope channel (P14/P10/P6 channel map, same as 260806XY/260812)
PADS = {"P14": 2, "P10": 3, "P6": 4}

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)
THRESHOLD_MV = -25.0


def per_event_ph(files):
    b_lo, b_hi = BASELINE_T_RANGE
    s_lo, s_hi = SIGNAL_T_RANGE
    raw = np.empty(len(files))
    for i, f in enumerate(files):
        with np.load(f) as z:
            t, v = z["t"], z["v"]
        b = np.median(v[(t >= b_lo) & (t <= b_hi)])
        sig = (t >= s_lo) & (t <= s_hi)
        raw[i] = (v[sig] - b).min()
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    args = ap.parse_args()
    cache_dir = os.path.join(args.cache_root, RUN)

    thr = -THRESHOLD_MV
    fig, axes = plt.subplots(1, len(PADS), figsize=(5.8 * len(PADS), 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    # Pass 1: load every pad's data first, so all panels can share one x-axis
    # range and one set of bin edges.
    loaded = []
    for pad, ch in PADS.items():
        files = sorted(glob.glob(os.path.join(cache_dir, f"{PREFIX}_ch{ch}_*.npz")))
        raw = per_event_ph(files)
        own_pass_idx = np.where(raw <= THRESHOLD_MV)[0]
        rail_val, rail_mask = detect_rail(raw[own_pass_idx]) if own_pass_idx.size else (None, np.array([], bool))
        n_clipped = int(rail_mask.sum()) if rail_val is not None else 0
        loaded.append((pad, ch, -raw, n_clipped, rail_val))

    hi = max(v.max() for _, _, v, _, _ in loaded)
    bin_width = 10.0
    n_bins = max(20, int(round(hi / bin_width)))

    for i, (pad, ch, vals_all, n_clipped, rail_val) in enumerate(loaded):
        color = SLOT_COLORS[i % len(SLOT_COLORS)]
        ax = axes[i]
        ax.set_facecolor(SURFACE)

        signal = vals_all[vals_all >= thr]
        frac_above = len(signal) / len(vals_all)

        counts, edges = np.histogram(vals_all, bins=n_bins, range=(0, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=color, alpha=0.85,
               edgecolor="none", zorder=3)
        ax.set_xlim(0, hi)  # shared across all panels for direct comparison

        signal_bin_max = counts[edges[:-1] >= thr].max() if (edges[:-1] >= thr).any() else counts.max()
        ax.set_ylim(0, max(signal_bin_max * 1.12, 5))
        ax.axvline(thr, color=MUTED, linewidth=1.1, linestyle="-", zorder=4)

        ax.set_title(f"{pad} (ch{ch}), X={X_DISP} mm (P14/P10 boundary)\n"
                     f"n={len(vals_all)} total, {len(signal)} ({100*frac_above:.1f}%) above threshold",
                     color=INK, fontsize=10.5, linespacing=1.4)
        ax.set_xlabel("per-event pulse height (mV)", color=INK, fontsize=10)
        ax.set_ylabel("events", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=SECONDARY, labelsize=8.5)

        print(f"[260814] {pad} (ch{ch}) X={X_DISP}mm: n_total={len(vals_all)} n_signal={len(signal)} "
              f"frac>={thr:.0f}mV={frac_above:.3f} n_clipped={n_clipped}"
              f"{f' clip_value={rail_val:.1f}mV' if rail_val is not None else ''}")

    fig.suptitle(f"{RUN}: per-event pulse-height distribution at the P14/P10 boundary (X={X_DISP}mm)",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = "hrppd_ph_dist_260814_boundary.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"[260814] wrote {out}")


if __name__ == "__main__":
    main()
