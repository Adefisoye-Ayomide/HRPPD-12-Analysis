"""
plot_raw_waveforms.py
======================
Stacked (overlaid) raw waveform plots, one figure per pad, at the X position
where that pad shows its peak mean pulse height (read from a results CSV
produced by analyze_hrppd.py, which also carries the pad -> channel mapping
used for that run). Lets you visually check the baseline window and
true-pulse window used by analyze_hrppd.py, e.g. confirming afterpulses land
outside the signal window.

All plots for one run share a common y-axis range so peak amplitudes are
directly comparable across pads.

Usage:
    python plot_raw_waveforms.py --results-csv hrppd_xscan_results_<run>.csv \\
        --cache-root ROOT --run RUN [--n-events N]
"""

import os, glob, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE   = (40.0, 65.0)

INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def half_max_centre(xs, ys):
    """
    Pad centre via half-max midpoint, NOT argmax: the response has a flat
    top (several mm within a few % of the true peak, see the pulse-height-
    vs-position plots), so argmax picks up whichever point happened to have
    the largest positive fluctuation on that flat top -- a real but noisy
    draw, not the geometric center. Half-max crossings (linearly
    interpolated) on either side of the peak are far less sensitive to
    single-point noise, since they're set by the whole rising/falling edge,
    not one sample. Returns (centre_x, lo_x, hi_x) or None if too few points.
    xs, ys must already be floor-subtracted and sorted by xs.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if xs.size < 3:
        return None
    peak = np.nanmax(ys)
    if not np.isfinite(peak) or peak <= 0:
        return None
    half = 0.5 * peak
    above = np.where(ys >= half)[0]
    if above.size == 0:
        return None

    def crossing(i, j):
        if j < 0 or j >= xs.size:
            return xs[i]
        dy = ys[i] - ys[j]
        return xs[i] if abs(dy) < 1e-12 else xs[i] + (ys[i] - half) / dy * (xs[j] - xs[i])

    lo = crossing(above[0], above[0] - 1)
    hi = crossing(above[-1], above[-1] + 1)
    return 0.5 * (lo + hi), lo, hi


def find_peak_positions(results_csv, min_n_frac=0.0):
    """
    For each pad, return (x, channel) nearest the half-max-midpoint centre
    of its (noise-floor-subtracted) response -- see half_max_centre().
    x is rounded to the nearest position actually present in the CSV (1mm
    grid).

    NOTE: unlike the old argmax version, this does NOT pre-filter positions
    by n_selected. n_selected is gate-driven (how many events had ANY pad
    fire), not a measure of this pad's own data quality -- filtering on it
    can silently DELETE the true peak region for a pad whose own hit rate is
    low even in-position (see hrppd_diag_P12_hitrate note), which then
    corrupts the half-max search far worse than a few noisy points would.
    min_n_frac is kept as an opt-in escape hatch (0 = off, the default).
    """
    rows_by_pad = {}  # pad -> list of (x, mean, n_selected, ctrl_mean)
    channel = {}       # pad -> channel
    pad_order = []
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            pad = row["pad"]
            if pad not in pad_order:
                pad_order.append(pad)
                rows_by_pad[pad] = []
            channel[pad] = int(row["channel"])
            rows_by_pad[pad].append((
                int(row["x_mm"]), float(row["mean_pulse_height_mV"]), int(row["n_selected"]),
                float(row["ctrl_mean_mV"]) if row.get("ctrl_mean_mV") not in (None, "") else 0.0,
            ))

    peak_x = {}
    for pad, rows in rows_by_pad.items():
        rows = sorted(rows, key=lambda r: r[0])
        max_n = max(n for _, _, n, _ in rows)
        trustworthy = [(x, mean, ctrl) for x, mean, n, ctrl in rows if n >= min_n_frac * max_n]
        xs = [t[0] for t in trustworthy]
        ys = [t[1] - t[2] for t in trustworthy]  # floor-subtracted
        result = half_max_centre(xs, ys)
        if result is None:
            # fall back to argmax if the response never reaches a clean half-max shape
            peak_x[pad] = max(trustworthy, key=lambda t: t[1])[0]
            continue
        centre, lo, hi = result
        nearest_x = min(xs, key=lambda x: abs(x - centre))
        peak_x[pad] = nearest_x
        print(f"[plot]   {pad}: half-max centre={centre:.2f}mm (edges {lo:.2f}->{hi:.2f}mm, "
              f"FWHM={hi-lo:.2f}mm) -> using nearest available X={nearest_x}mm")
    return pad_order, peak_x, channel


def main():
    ap = argparse.ArgumentParser(description="Stacked raw waveform plots per pad.")
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-events", type=int, default=1000,
                     help="number of waveforms to overlay per pad (default 1000)")
    ap.add_argument("--out-prefix", default=None,
                     help="default: hrppd_waveforms_<run>")
    args = ap.parse_args()
    out_prefix = args.out_prefix or f"hrppd_waveforms_{args.run}"

    cache_dir = os.path.join(args.cache_root, args.run)
    pad_order, peak_x, channel = find_peak_positions(args.results_csv)
    pad_color = {pad: SLOT_COLORS[i % len(SLOT_COLORS)] for i, pad in enumerate(pad_order)}
    print(f"[plot] peak positions from {args.results_csv}: {peak_x}")

    all_v = {}
    for pad in pad_order:
        x, ch = peak_x[pad], channel[pad]
        files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))[:args.n_events]
        if not files:
            print(f"[plot] no files for ch{ch} ({pad}) at X={x}, skipping")
            continue
        ts, vs = [], []
        for f in files:
            with np.load(f) as z:
                ts.append(z["t"])
                vs.append(z["v"])
        all_v[pad] = (x, ch, ts, vs)

    y_min = min(v.min() for _, _, _, vs in all_v.values() for v in vs)
    y_max = max(v.max() for _, _, _, vs in all_v.values() for v in vs)
    pad_margin = 0.05 * (y_max - y_min)
    y_range = (y_min - pad_margin, y_max + pad_margin)

    for pad in pad_order:
        if pad not in all_v:
            continue
        x, ch, ts, vs = all_v[pad]
        color = pad_color[pad]

        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        for t, v in zip(ts, vs):
            ax.plot(t, v, color=color, alpha=0.05, linewidth=0.8, zorder=2)

        ax.axvspan(*BASELINE_T_RANGE, color=MUTED, alpha=0.15, zorder=1,
                    label=f"baseline window [{BASELINE_T_RANGE[0]:.0f}, {BASELINE_T_RANGE[1]:.0f}] ns")
        ax.axvspan(*SIGNAL_T_RANGE, color=color, alpha=0.12, zorder=1,
                    label=f"signal window [{SIGNAL_T_RANGE[0]:.0f}, {SIGNAL_T_RANGE[1]:.0f}] ns")

        ax.set_ylim(*y_range)
        ax.set_xlabel("Time (ns)", color=INK, fontsize=11)
        ax.set_ylabel("Voltage (mV)", color=INK, fontsize=11)
        ax.set_title(f"{pad} (ch{ch}) raw waveforms at X={x} mm  (n={len(ts)} stacked)",
                      color=INK, fontsize=12, pad=12)

        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=SECONDARY, labelsize=9)

        legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
        for text in legend.get_texts():
            text.set_color(INK)

        fig.tight_layout()
        out = f"{out_prefix}_{pad}.png"
        fig.savefig(out, facecolor=SURFACE)
        plt.close(fig)
        print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
