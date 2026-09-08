"""
plot_ph_distribution.py
========================
Per-event pulse-height distribution at each pad's peak position (the same
X used by plot_raw_waveforms.py), one histogram per pad. Complements the
raw waveform overlay: that shows what the traces look like, this shows the
actual per-event PH numbers that feed the mean-pulse-height-vs-position
curve -- reveals shape (single-PE-like broad/skewed tail vs. clean peaked
distribution), any residual clipping spike, and whether "mean" is a
representative summary of that shape.

Usage:
    python plot_ph_distribution.py --results-csv hrppd_xscan_results_<run>.csv \\
        --cache-root ROOT --run RUN
"""

import os, glob, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_raw_waveforms import find_peak_positions, CACHE_ROOT
from analyze_hrppd import detect_rail

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)


def per_event_ph(cache_dir, x, ch, threshold_mv):
    """
    Returns (vals, n_clipped, clip_value). vals is positive-magnitude PH for
    EVERY event -- clipped/railed events are NOT removed, only detected and
    counted (via the same detect_rail() used by analyze_hrppd.py) so this is
    genuinely all the data, same as the CSV's mean would be without the
    clip-exclusion analyze_hrppd.py applies. n_clipped/clip_value are
    reported for reference only.
    """
    files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))
    raw = np.empty(len(files))
    b_lo, b_hi = BASELINE_T_RANGE
    s_lo, s_hi = SIGNAL_T_RANGE
    for i, f in enumerate(files):
        with np.load(f) as z:
            t, v = z["t"], z["v"]
        b = np.median(v[(t >= b_lo) & (t <= b_hi)])
        sig = (t >= s_lo) & (t <= s_hi)
        raw[i] = (v[sig] - b).min()  # negative-going raw value, matches analyze_hrppd.py's convention

    own_pass_idx = np.where(raw <= threshold_mv)[0]
    rail_val, rail_mask = detect_rail(raw[own_pass_idx]) if own_pass_idx.size else (None, np.array([], bool))
    n_clipped = int(rail_mask.sum()) if rail_val is not None else 0
    return -raw, n_clipped, rail_val


def main():
    ap = argparse.ArgumentParser(description="Per-event pulse-height distribution at each pad's peak.")
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=float, default=-25.0)
    ap.add_argument("--out-prefix", default=None, help="default: hrppd_ph_dist_<run>")
    args = ap.parse_args()
    out_prefix = args.out_prefix or f"hrppd_ph_dist_{args.run}"

    cache_dir = os.path.join(args.cache_root, args.run)
    pad_order, peak_x, channel = find_peak_positions(args.results_csv)
    thr = -args.threshold  # positive-magnitude threshold, e.g. 25.0

    # Pass 1: load every pad's data first, so all panels can share one x-axis
    # range and one set of bin edges -- same mV windows in every panel, not
    # just visually-similar-looking ones.
    loaded = []
    for pad in pad_order:
        x, ch = peak_x[pad], channel[pad]
        vals_all, n_clipped, clip_val = per_event_ph(cache_dir, x, ch, args.threshold)
        loaded.append((pad, x, ch, vals_all, n_clipped, clip_val))

    hi = max(v.max() for _, _, _, v, _, _ in loaded)  # shared across all panels
    bin_width = 10.0  # mV, per advisor's request
    n_bins = max(20, int(round(hi / bin_width)))

    fig, axes = plt.subplots(1, len(pad_order), figsize=(5.6 * len(pad_order), 4.8), dpi=150)
    if len(pad_order) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for i, (pad, x, ch, vals_all, n_clipped, clip_val) in enumerate(loaded):
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

        # y-limit pinned close to the tallest SIGNAL-region bin (little headroom),
        # so a secondary spike -- e.g. a clipping pileup -- fills the frame and
        # reads as the dominant feature it is, while the noise peak is still cut.
        signal_bin_max = counts[edges[:-1] >= thr].max() if (edges[:-1] >= thr).any() else counts.max()
        ax.set_ylim(0, max(signal_bin_max * 1.12, 5))

        ax.axvline(thr, color=MUTED, linewidth=1.1, linestyle="-", zorder=4)

        noise_peak = counts.max()

        ax.set_title(f"{pad} (ch{ch}), X={x} mm\nn={len(vals_all)} total, "
                     f"{len(signal)} ({100*frac_above:.1f}%) above threshold",
                     color=INK, fontsize=10.5, linespacing=1.4)
        ax.set_xlabel("per-event pulse height (mV)", color=INK, fontsize=10)
        ax.set_ylabel("events", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=SECONDARY, labelsize=8.5)

        print(f"[ph_dist] {pad} x={x}mm n_total={len(vals_all)} n_signal={len(signal)} "
              f"frac>={thr:.0f}mV={frac_above:.3f} noise_peak_count={noise_peak} bin_width={bin_width}mV n_bins={n_bins}"
              f"{f' clipped={n_clipped}@{clip_val:.0f}mV' if n_clipped else ''}")

    fig.suptitle(f"{args.run}: per-event pulse-height distribution at each pad's peak position",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = f"{out_prefix}.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"[ph_dist] wrote {out}")


if __name__ == "__main__":
    main()
