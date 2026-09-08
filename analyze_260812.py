"""
analyze_260812.py
==================
Dedicated single-position verification run, one capture per pad at its own
peak position (not a full X-scan): checks whether raising the threshold to
-25 mV (and whatever hardware change prompted re-taking this) has actually
fixed clipping, and what the per-event pulse-height distribution looks like
at full statistics for each pad.

FILENAME NOTE: the run was taken with a labelling mistake -- the file
prefix for the ch3/P10 capture is "AA0812X23", but the laser was actually
sitting at P10's true peak position (40 mm, matching 260806XY). The prefix
is used only to find the files; "40" (the true position) is what's used for
display/labels everywhere in this script's output.

Produces, per pad: a representative raw-waveform overlay, and a per-event
pulse-height histogram (all events, no cuts, adaptive binning) with clip
detection -- same conventions as plot_raw_waveforms.py / plot_ph_distribution.py.
"""

import os, glob, argparse
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_hrppd import detect_rail

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
RUN = "260812"

# pad label -> (file prefix as actually written to disk, scope channel, true/display X in mm)
CAPTURES = {
    "P14": ("AA0812X17", 2, 17),
    "P10": ("AA0812X23", 3, 40),  # mislabelled prefix; true position is 40mm
    "P6":  ("AA0812X67", 4, 67),
}

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)
THRESHOLD_MV = -25.0


def load_events(cache_dir, prefix, ch, n_max=None):
    files = sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_ch{ch}_*.npz")))
    if n_max:
        files = files[:n_max]
    return files


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
    return raw  # negative-going, mV


def plot_waveforms(cache_dir, out_path, n_overlay=2000):
    fig, axes = plt.subplots(1, len(CAPTURES), figsize=(6.2 * len(CAPTURES), 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for i, (pad, (prefix, ch, x_disp)) in enumerate(CAPTURES.items()):
        color = SLOT_COLORS[i % len(SLOT_COLORS)]
        ax = axes[i]
        ax.set_facecolor(SURFACE)

        files = load_events(cache_dir, prefix, ch)
        sample_idx = np.linspace(0, len(files) - 1, min(n_overlay, len(files))).astype(int)

        b_lo, b_hi = BASELINE_T_RANGE
        for j in sample_idx:
            with np.load(files[j]) as z:
                t, v = z["t"], z["v"]
            b = np.median(v[(t >= b_lo) & (t <= b_hi)])
            m = (t >= 20) & (t <= 90)
            ax.plot(t[m], (v - b)[m], color=color, alpha=0.02, linewidth=0.7, zorder=2)

        ax.axvspan(*SIGNAL_T_RANGE, color=color, alpha=0.10, zorder=1,
                   label=f"signal window {SIGNAL_T_RANGE} ns")
        ax.axhline(THRESHOLD_MV, color=MUTED, linewidth=1, linestyle="--", zorder=3)

        ax.set_title(f"{pad} (ch{ch}), X={x_disp} mm  --  n={len(files)}", color=INK, fontsize=12)
        ax.set_xlabel("Time (ns)", color=INK, fontsize=10)
        ax.set_ylabel("baseline-subtracted voltage (mV)", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=SECONDARY, labelsize=9)
        legend = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        for t in legend.get_texts():
            t.set_color(INK)

    fig.suptitle(f"{RUN}: raw waveforms at each pad's peak position ({n_overlay} shots overlaid)",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"[260812] wrote {out_path}")


def plot_ph_and_clip(cache_dir, out_path):
    fig, axes = plt.subplots(1, len(CAPTURES), figsize=(5.8 * len(CAPTURES), 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    thr = -THRESHOLD_MV

    # Pass 1: load every pad's data first, so all panels can share one x-axis
    # range and one set of bin edges.
    loaded = []
    for pad, (prefix, ch, x_disp) in CAPTURES.items():
        files = load_events(cache_dir, prefix, ch)
        raw = per_event_ph(files)
        own_pass_idx = np.where(raw <= THRESHOLD_MV)[0]
        rail_val, rail_mask = detect_rail(raw[own_pass_idx]) if own_pass_idx.size else (None, np.array([], bool))
        n_clipped = int(rail_mask.sum()) if rail_val is not None else 0
        loaded.append((pad, ch, x_disp, -raw, n_clipped, rail_val))

    hi = max(v.max() for _, _, _, v, _, _ in loaded)
    bin_width = 10.0
    n_bins = max(20, int(round(hi / bin_width)))

    for i, (pad, ch, x_disp, vals_all, n_clipped, rail_val) in enumerate(loaded):
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

        ax.set_title(f"{pad} (ch{ch}), X={x_disp} mm\nn={len(vals_all)} total, "
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

        print(f"[260812] {pad} (ch{ch}) X={x_disp}mm: n_total={len(vals_all)} n_signal={len(signal)} "
              f"frac>={thr:.0f}mV={frac_above:.3f} n_clipped={n_clipped}"
              f"{f' clip_value={rail_val:.1f}mV' if rail_val is not None else ''}")

    fig.suptitle(f"{RUN}: per-event pulse-height distribution at each pad's peak position",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"[260812] wrote {out_path}")


def plot_grid(cache_dir, out_path):
    """
    3x3: rows = laser position (each pad's own peak), columns = pad/channel.
    The diagonal reproduces plot_ph_and_clip()'s per-pad distributions; the
    off-diagonal panels show what a NEIGHBORING pad's own signal looks like
    while the laser sits on someone else's peak -- pure pedestal if there's
    no crosstalk/charge-sharing, a bump under the pedestal if there is.
    """
    thr = -THRESHOLD_MV
    pads = list(CAPTURES.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for row, (laser_pad, (prefix, _laser_ch, x_disp)) in enumerate(CAPTURES.items()):
        for col, pad in enumerate(pads):
            ch = CAPTURES[pad][1]
            color = SLOT_COLORS[col % len(SLOT_COLORS)]
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)

            files = load_events(cache_dir, prefix, ch)
            raw = per_event_ph(files)
            vals_all = -raw
            signal = vals_all[vals_all >= thr]
            frac_above = len(signal) / len(vals_all)

            hi = 500.0
            bin_width = 10.0
            n_bins = int(round(hi / bin_width))
            counts, edges = np.histogram(vals_all, bins=n_bins, range=(0, hi))
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=color, alpha=0.85,
                   edgecolor="none", zorder=3)
            ax.set_xlim(0, hi)  # shared across all 9 panels for direct comparison

            signal_bin_max = counts[edges[:-1] >= thr].max() if (edges[:-1] >= thr).any() else counts.max()
            ax.set_ylim(0, max(signal_bin_max * 1.12, 5))
            ax.axvline(thr, color=MUTED, linewidth=1.1, linestyle="-", zorder=4)

            tag = "own peak" if pad == laser_pad else "neighbor"
            ax.set_title(f"laser@{laser_pad} (X={x_disp}mm)  |  reading: {pad} (ch{ch})  [{tag}]\n"
                         f"n={len(vals_all)}, {len(signal)} ({100*frac_above:.1f}%) above threshold",
                         color=INK, fontsize=9.5, linespacing=1.35)
            ax.set_xlabel("pulse height (mV)", color=INK, fontsize=9)
            ax.set_ylabel("events", color=INK, fontsize=9)
            ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(MUTED)
            ax.tick_params(colors=SECONDARY, labelsize=8)

            print(f"[260812/grid] laser@{laser_pad} reading {pad}(ch{ch}): "
                  f"n={len(vals_all)} frac>={thr:.0f}mV={frac_above:.4f}")

    fig.suptitle(f"{RUN}: all 3 pads' response at each of the 3 laser positions (9 distributions)",
                 color=INK, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"[260812] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    args = ap.parse_args()
    cache_dir = os.path.join(args.cache_root, RUN)

    plot_waveforms(cache_dir, "hrppd_waveforms_260812.png")
    plot_ph_and_clip(cache_dir, "hrppd_ph_dist_260812.png")
    plot_grid(cache_dir, "hrppd_ph_dist_260812_grid.png")


if __name__ == "__main__":
    main()
