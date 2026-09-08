"""
plot_260819XY_peaks_boundaries.py
====================================
Peak and boundary waveform-overlay + pulse-height-distribution plots for the
260819XY run (P9=ch2, P5=ch3, P1=ch4 -- "the other edge", row 2).

Peak positions (from hrppd_xscan_results_260819XY.csv), all real, clean
single peaks within the scanned range:
    P9 -> X=11mm (162.9 mV)
    P5 -> X=42mm (162.1 mV)
    P1 -> X=61mm (152.5 mV)

Boundaries (curve crossings):
    P9/P5 -> X=27mm
    P5/P1 -> X=53mm

Same conventions as the rest of the project: full-event overlay (no
subsampling) with shared y-axis per figure for waveforms; 10mV-bin,
shared-x-axis, no-cut pulse-height distributions.
"""

import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 20000
import matplotlib.pyplot as plt

from analyze_hrppd import detect_rail
from plot_waveform_overlays import plot_overlay

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
RUN = "260819XY"
OUT_DIR = "/home/matthew/HRPPD_characterization/overlay_plots"

CHANNELS = {"P9": 2, "P5": 3, "P1": 4}
SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)
THRESHOLD_MV = -25.0

PEAKS = {"P9": 11, "P5": 42, "P1": 61}
BOUNDARIES = [(27, "P9/P5"), (53, "P5/P1")]


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


def plot_ph_dist(cache_dir, panels, suptitle, out_path):
    """panels: list of (label, ch, x_disp)"""
    thr = -THRESHOLD_MV
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.8), dpi=150)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    loaded = []
    for label, ch, x_disp in panels:
        files = sorted(glob.glob(os.path.join(cache_dir, f"AA0819X{x_disp}Y0_ch{ch}_*.npz")))
        raw = per_event_ph(files)
        vals_all = -raw
        own_pass_idx = np.where(raw <= THRESHOLD_MV)[0]
        rail_val, rail_mask = detect_rail(raw[own_pass_idx]) if own_pass_idx.size else (None, np.array([], bool))
        n_clipped = int(rail_mask.sum()) if rail_val is not None else 0
        loaded.append((label, ch, x_disp, vals_all, n_clipped, rail_val))

    hi = max(v.max() for *_, v, _, _ in loaded)
    bin_width = 10.0
    n_bins = max(20, int(round(hi / bin_width)))

    for i, (label, ch, x_disp, vals_all, n_clipped, rail_val) in enumerate(loaded):
        color = SLOT_COLORS[i % len(SLOT_COLORS)]
        ax = axes[i]
        ax.set_facecolor(SURFACE)

        signal = vals_all[vals_all >= thr]
        frac_above = len(signal) / len(vals_all)

        counts, edges = np.histogram(vals_all, bins=n_bins, range=(0, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=color, alpha=0.85,
               edgecolor="none", zorder=3)
        ax.set_xlim(0, hi)

        signal_bin_max = counts[edges[:-1] >= thr].max() if (edges[:-1] >= thr).any() else counts.max()
        ax.set_ylim(0, max(signal_bin_max * 1.12, 5))
        ax.axvline(thr, color=MUTED, linewidth=1.1, linestyle="-", zorder=4)

        ax.set_title(f"{label} (ch{ch}), X={x_disp} mm\n"
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

        print(f"[260819XY] {label} (ch{ch}) X={x_disp}mm: n_total={len(vals_all)} n_signal={len(signal)} "
              f"frac>={thr:.0f}mV={frac_above:.3f} n_clipped={n_clipped}"
              f"{f' clip_value={rail_val:.1f}mV' if rail_val is not None else ''}")

    fig.suptitle(suptitle, color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[260819XY] wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(CACHE_ROOT, RUN)

    # ---- peaks: waveform overlay ----
    plot_overlay(
        cache_dir,
        [("P9", f"AA0819X{PEAKS['P9']}Y0", 2, PEAKS["P9"]),
         ("P5", f"AA0819X{PEAKS['P5']}Y0", 3, PEAKS["P5"]),
         ("P1", f"AA0819X{PEAKS['P1']}Y0", 4, PEAKS["P1"])],
        "260819XY: raw waveforms at each pad's peak (all shots overlaid)",
        os.path.join(OUT_DIR, "waveforms_260819XY_peak.png"),
    )

    # ---- peaks: pulse-height distribution ----
    plot_ph_dist(
        cache_dir,
        [("P9", 2, PEAKS["P9"]), ("P5", 3, PEAKS["P5"]), ("P1", 4, PEAKS["P1"])],
        "260819XY: per-event pulse-height distribution at each pad's peak",
        os.path.join(OUT_DIR, "ph_dist_260819XY_peak.png"),
    )

    # ---- boundaries ----
    for x_disp, label in BOUNDARIES:
        tag = label.replace("/", "")
        plot_overlay(
            cache_dir,
            [("P9", f"AA0819X{x_disp}Y0", 2, x_disp),
             ("P5", f"AA0819X{x_disp}Y0", 3, x_disp),
             ("P1", f"AA0819X{x_disp}Y0", 4, x_disp)],
            f"260819XY: raw waveforms at the {label} boundary (X={x_disp}mm, all shots overlaid)",
            os.path.join(OUT_DIR, f"waveforms_260819XY_boundary_{tag}.png"),
        )
        plot_ph_dist(
            cache_dir,
            [("P9", 2, x_disp), ("P5", 3, x_disp), ("P1", 4, x_disp)],
            f"260819XY: per-event pulse-height distribution at the {label} boundary (X={x_disp}mm)",
            os.path.join(OUT_DIR, f"ph_dist_260819XY_boundary_{tag}.png"),
        )


if __name__ == "__main__":
    main()
