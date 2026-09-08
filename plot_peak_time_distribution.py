"""
plot_peak_time_distribution.py
================================
Distribution of the instantaneous per-event voltage at ONE FIXED TIME --
each pad's typical peak-arrival time, found from the mean of many baseline-
subtracted traces -- rather than each event's own minimum within the signal
window (that's what plot_ph_distribution.py shows).

Why this is a different, useful view: pulse-height (per-event min over the
whole signal window) never misses a jittered pulse, since it searches the
full window per event. Freezing time at one fixed instant instead means an
event whose true pulse arrives a few ns early or late will show a SMALLER
value here than its own true peak -- so this distribution mixes real
amplitude spread with timing jitter. Comparing the two plots side by side is
itself informative: if they look very similar, timing jitter isn't smearing
the amplitude picture much; if this one is visibly broader/lower, jitter
matters.

The peak time is not assumed -- it's found per pad from where the MEAN
trace (averaged over many events, which washes out per-event noise) dips
lowest, and printed/used per pad rather than a single hardcoded value for
every pad.

Usage:
    python plot_peak_time_distribution.py --results-csv hrppd_xscan_results_<run>.csv \\
        --cache-root ROOT --run RUN
"""

import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_raw_waveforms import find_peak_positions, CACHE_ROOT

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)


def find_peak_time(cache_dir, x, ch, n_sample=1000):
    """Mean baseline-subtracted trace across n_sample events -> time of its minimum."""
    files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))[:n_sample]
    b_lo, b_hi = BASELINE_T_RANGE
    s_lo, s_hi = SIGNAL_T_RANGE
    acc, t_ref, n = None, None, 0
    for f in files:
        with np.load(f) as z:
            t, v = z["t"], z["v"]
        b = np.median(v[(t >= b_lo) & (t <= b_hi)])
        m = (t >= s_lo) & (t <= s_hi)
        vv = v[m] - b
        if t_ref is None:
            t_ref = t[m]
            acc = np.zeros_like(t_ref)
        if vv.shape != acc.shape:
            continue
        acc += vv
        n += 1
    mean_trace = acc / n
    i_min = int(np.argmin(mean_trace))
    return float(t_ref[i_min])


def per_event_value_at_time(cache_dir, x, ch, t_fixed):
    files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))
    vals = np.empty(len(files))
    b_lo, b_hi = BASELINE_T_RANGE
    for i, f in enumerate(files):
        with np.load(f) as z:
            t, v = z["t"], z["v"]
        b = np.median(v[(t >= b_lo) & (t <= b_hi)])
        idx = int(np.argmin(np.abs(t - t_fixed)))
        vals[i] = -(v[idx] - b)  # positive-magnitude, same convention as pulse height
    return vals


def main():
    ap = argparse.ArgumentParser(description="Per-event voltage distribution at a fixed peak time.")
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=float, default=-25.0)
    ap.add_argument("--out-prefix", default=None, help="default: hrppd_peaktime_dist_<run>")
    args = ap.parse_args()
    out_prefix = args.out_prefix or f"hrppd_peaktime_dist_{args.run}"

    cache_dir = os.path.join(args.cache_root, args.run)
    pad_order, peak_x, channel = find_peak_positions(args.results_csv)
    thr = -args.threshold

    # Pass 1: load every pad's data first, so all panels can share one x-axis
    # range and one set of bin edges.
    loaded = []
    for pad in pad_order:
        x, ch = peak_x[pad], channel[pad]
        t_peak = find_peak_time(cache_dir, x, ch)
        vals_all = per_event_value_at_time(cache_dir, x, ch, t_peak)
        loaded.append((pad, x, ch, t_peak, vals_all))

    hi = max(v.max() for _, _, _, _, v in loaded)
    lo = min(0, min(v.min() for _, _, _, _, v in loaded))
    bin_width = 3.5
    n_bins = max(20, int(round((hi - lo) / bin_width)))

    fig, axes = plt.subplots(1, len(pad_order), figsize=(5.6 * len(pad_order), 4.8), dpi=150)
    if len(pad_order) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for i, (pad, x, ch, t_peak, vals_all) in enumerate(loaded):
        color = SLOT_COLORS[i % len(SLOT_COLORS)]
        ax = axes[i]
        ax.set_facecolor(SURFACE)

        signal = vals_all[vals_all >= thr]
        frac_above = len(signal) / len(vals_all)

        counts, edges = np.histogram(vals_all, bins=n_bins, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=color, alpha=0.85,
               edgecolor="none", zorder=3)
        ax.set_xlim(lo, hi)  # shared across all panels for direct comparison

        signal_bin_max = counts[edges[:-1] >= thr].max() if (edges[:-1] >= thr).any() else counts.max()
        ax.set_ylim(0, max(signal_bin_max * 2.2, 5))

        ax.axvline(thr, color=MUTED, linewidth=1.1, linestyle="-", zorder=4)

        noise_peak = counts.max()

        ax.set_title(f"{pad} (ch{ch}), X={x} mm, t={t_peak:.1f} ns\n"
                     f"n={len(vals_all)} total, {len(signal)} ({100*frac_above:.1f}%) above threshold",
                     color=INK, fontsize=10.5, linespacing=1.4)
        ax.set_xlabel(f"voltage at t={t_peak:.1f} ns (mV)", color=INK, fontsize=10)
        ax.set_ylabel("events", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=SECONDARY, labelsize=8.5)

        print(f"[peaktime_dist] {pad} x={x}mm t_peak={t_peak:.2f}ns n_total={len(vals_all)} "
              f"n_signal={len(signal)} frac>={thr:.0f}mV={frac_above:.3f} noise_peak_count={noise_peak} "
              f"bin_width={bin_width}mV n_bins={n_bins}")

    fig.suptitle(f"{args.run}: per-event voltage at each pad's fixed peak time",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = f"{out_prefix}.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"[peaktime_dist] wrote {out}")


if __name__ == "__main__":
    main()
