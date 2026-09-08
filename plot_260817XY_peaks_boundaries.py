"""
plot_260817XY_peaks_boundaries.py
====================================
Peak and boundary waveform-overlay + pulse-height-distribution plots for the
260817XY run (P10=ch2, P6=ch3, P2=ch4 -- "the other edge").

Peak positions (from hrppd_xscan_results_260817XY.csv):
    P10 -> X=36mm (208.0 mV, clean peak)
    P6  -> X=61mm (199.6 mV, clean peak)
    P2  -> X=69mm (23.9 mV) -- NOT a real peak: P2 stays at noise floor
           (~8-10mV) across the whole 0-67mm scan and is still climbing at
           the last position. This is the best-available point, plotted at
           the user's explicit request, but is clearly labeled as likely
           incomplete pending a wider re-scan.

Boundary: P10/P6 curves cross between X=49 and X=50mm -> use X=50mm.
No P6/P2 boundary is plotted: P6 (~190mV) and P2 (~10-24mV) never cross
within the scanned range, so there is no real boundary to show.

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
from plot_waveform_overlays import plot_overlay, CACHE_ROOT as WF_CACHE_ROOT

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
RUN = "260817XY"
OUT_DIR = "/home/matthew/HRPPD_characterization/overlay_plots"

CHANNELS = {"P10": 2, "P6": 3, "P2": 4}
SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)
THRESHOLD_MV = -25.0

PEAKS = {"P10": 36, "P6": 61, "P2": 69}
BOUNDARY_X = 50


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
    """panels: list of (label, ch, x_disp, extra_title_tag)"""
    thr = -THRESHOLD_MV
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.8), dpi=150)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    loaded = []
    for label, ch, x_disp, tag in panels:
        files = sorted(glob.glob(os.path.join(cache_dir, f"AA0817X{x_disp}Y0_ch{ch}_*.npz")))
        raw = per_event_ph(files)
        vals_all = -raw
        own_pass_idx = np.where(raw <= THRESHOLD_MV)[0]
        rail_val, rail_mask = detect_rail(raw[own_pass_idx]) if own_pass_idx.size else (None, np.array([], bool))
        n_clipped = int(rail_mask.sum()) if rail_val is not None else 0
        loaded.append((label, ch, x_disp, tag, vals_all, n_clipped, rail_val))

    hi = max(v.max() for *_, v, _, _ in loaded)
    bin_width = 10.0
    n_bins = max(20, int(round(hi / bin_width)))

    for i, (label, ch, x_disp, tag, vals_all, n_clipped, rail_val) in enumerate(loaded):
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

        ax.set_title(f"{label} (ch{ch}), X={x_disp} mm{tag}\n"
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

        print(f"[260817XY] {label} (ch{ch}) X={x_disp}mm: n_total={len(vals_all)} n_signal={len(signal)} "
              f"frac>={thr:.0f}mV={frac_above:.3f} n_clipped={n_clipped}"
              f"{f' clip_value={rail_val:.1f}mV' if rail_val is not None else ''}")

    fig.suptitle(suptitle, color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[260817XY] wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(CACHE_ROOT, RUN)

    # ---- peaks: waveform overlay ----
    plot_overlay(
        cache_dir,
        [("P10", f"AA0817X{PEAKS['P10']}Y0", 2, PEAKS["P10"]),
         ("P6", f"AA0817X{PEAKS['P6']}Y0", 3, PEAKS["P6"]),
         ("P2", f"AA0817X{PEAKS['P2']}Y0", 4, PEAKS["P2"])],
        "260817XY: raw waveforms at each pad's peak (all shots overlaid)\n"
        "P2 @ X=69mm is the highest point in-range, NOT a confirmed peak -- still rising at scan edge",
        os.path.join(OUT_DIR, "waveforms_260817XY_peak.png"),
    )

    # ---- peaks: pulse-height distribution ----
    plot_ph_dist(
        cache_dir,
        [("P10", 2, PEAKS["P10"], ""), ("P6", 3, PEAKS["P6"], ""),
         ("P2", 4, PEAKS["P2"], " (best-available, likely incomplete)")],
        "260817XY: per-event pulse-height distribution at each pad's peak",
        os.path.join(OUT_DIR, "ph_dist_260817XY_peak.png"),
    )

    # ---- P10/P6 boundary: waveform overlay ----
    plot_overlay(
        cache_dir,
        [("P10", f"AA0817X{BOUNDARY_X}Y0", 2, BOUNDARY_X),
         ("P6", f"AA0817X{BOUNDARY_X}Y0", 3, BOUNDARY_X),
         ("P2", f"AA0817X{BOUNDARY_X}Y0", 4, BOUNDARY_X)],
        f"260817XY: raw waveforms at the P10/P6 boundary (X={BOUNDARY_X}mm, all shots overlaid)",
        os.path.join(OUT_DIR, "waveforms_260817XY_boundary_P10P6.png"),
    )

    # ---- P10/P6 boundary: pulse-height distribution ----
    plot_ph_dist(
        cache_dir,
        [("P10", 2, BOUNDARY_X, ""), ("P6", 3, BOUNDARY_X, ""), ("P2", 4, BOUNDARY_X, "")],
        f"260817XY: per-event pulse-height distribution at the P10/P6 boundary (X={BOUNDARY_X}mm)",
        os.path.join(OUT_DIR, "ph_dist_260817XY_boundary_P10P6.png"),
    )


if __name__ == "__main__":
    main()
