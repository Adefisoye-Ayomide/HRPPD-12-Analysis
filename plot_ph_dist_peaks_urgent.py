import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_hrppd import detect_rail

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
OUT_DIR = "/home/matthew/HRPPD_characterization/overlay_plots"
os.makedirs(OUT_DIR, exist_ok=True)

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


def plot_peak_dist(run, prefix_fmt, panels, out_path, suptitle):
    """panels: list of (label, ch, x_disp, tag)"""
    thr = -THRESHOLD_MV
    cache_dir = os.path.join(CACHE_ROOT, run)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.8), dpi=150)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    loaded = []
    for label, ch, x_disp, tag in panels:
        files = sorted(glob.glob(os.path.join(cache_dir, prefix_fmt.format(x=x_disp) + f"_ch{ch}_*.npz")))
        raw = per_event_ph(files)
        vals_all = -raw
        own_pass_idx = np.where(raw <= THRESHOLD_MV)[0]
        rail_val, rail_mask = detect_rail(raw[own_pass_idx]) if own_pass_idx.size else (None, np.array([], bool))
        n_clipped = int(rail_mask.sum()) if rail_val is not None else 0
        loaded.append((label, ch, x_disp, tag, vals_all, n_clipped, rail_val))

    hi = max(v.max() for *_, v, _, _ in loaded)
    n_bins = max(20, int(round(hi / 10.0)))

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
        print(f"[{run}] {label} ch{ch} X={x_disp}: n={len(vals_all)} sig={len(signal)} clip={n_clipped}")

    fig.suptitle(suptitle, color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[{run}] wrote {out_path}")


# 260821XY peaks: P9@8, P5@36, P1@61
plot_peak_dist(
    "260821XY", "AA0821X{x}Y0",
    [("P9", 2, 8, ""), ("P5", 3, 36, ""), ("P1", 4, 61, "")],
    os.path.join(OUT_DIR, "ph_dist_260821XY_peak.png"),
    "260821XY: per-event pulse-height distribution at each pad's peak",
)

# 260822XY peaks: P11 has NO real peak (best-available X=8, flagged), P7@49, P3@68
plot_peak_dist(
    "260822XY", "AA0822X{x}Y0",
    [("P11", 2, 8, " (best-available, likely no real peak in-range)"), ("P7", 3, 49, ""), ("P3", 4, 68, "")],
    os.path.join(OUT_DIR, "ph_dist_260822XY_peak.png"),
    "260822XY: per-event pulse-height distribution at each pad's peak",
)
