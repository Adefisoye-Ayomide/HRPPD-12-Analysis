"""
plot_waveform_overlays.py
===========================
Overlaid raw-waveform plots (baseline-subtracted, ALL shots translucent) for
peak and boundary positions across the 260812 (P14/P10/P6) and 260814XY
(P13/P9/P5) datasets, all collected in one folder for easy side-by-side
comparison. Same visual conventions as analyze_260812.py's plot_waveforms():
signal window shaded, threshold line dashed, one panel per channel -- plus
every panel within a figure shares one y-axis (set from the deepest dip
across that figure's panels) so amplitudes are directly comparable.

Every event is drawn (not a subsample); panels are rendered as a single
NaN-separated line for speed rather than one ax.plot() call per event.

Usage:
    python3 plot_waveform_overlays.py
"""

import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 20000  # avoid Agg cell-block overflow on huge overlaid paths
import matplotlib.pyplot as plt

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
OUT_DIR = "/home/matthew/HRPPD_characterization/overlay_plots"

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#525142", "#898781", "#e1e0d9", "#fcfcfb"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)
THRESHOLD_MV = -25.0
DISPLAY_T_RANGE = (20.0, 90.0)
TARGET_ALPHA_COVERAGE = 40.0  # ~ n_lines * alpha, tuned so dense regions read solid


def load_panel(cache_dir, prefix, ch):
    b_lo, b_hi = BASELINE_T_RANGE
    d_lo, d_hi = DISPLAY_T_RANGE
    files = sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_ch{ch}_*.npz")))

    t_chunks, v_chunks = [], []
    vmin = 0.0
    for f in files:
        with np.load(f) as z:
            t, v = z["t"], z["v"]
        b = np.median(v[(t >= b_lo) & (t <= b_hi)])
        m = (t >= d_lo) & (t <= d_hi)
        tm, vm = t[m], (v - b)[m]
        vmin = min(vmin, vm.min())
        # NaN separator so the whole panel is one Line2D (fast to render)
        t_chunks.append(np.append(tm, np.nan))
        v_chunks.append(np.append(vm, np.nan))

    tx = np.concatenate(t_chunks) if t_chunks else np.array([])
    vy = np.concatenate(v_chunks) if v_chunks else np.array([])
    return tx, vy, len(files), vmin


def plot_overlay(cache_dir, panels, suptitle, out_path):
    """
    panels: list of (panel_label, prefix, channel, x_disp) in display order.
    All panels in the figure share one y-axis, set by the deepest dip
    among them.
    """
    loaded = []
    for label, prefix, ch, x_disp in panels:
        tx, vy, n, vmin = load_panel(cache_dir, prefix, ch)
        loaded.append((label, prefix, ch, x_disp, tx, vy, n, vmin))
        print(f"[overlay] loaded {label} (ch{ch}) X={x_disp}mm: n={n}, min={vmin:.1f}mV")

    y_lo = min(v[7] for v in loaded) * 1.08  # shared across all panels, small headroom
    y_hi = max(20.0, y_lo * -0.05)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2), dpi=150)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for i, (label, prefix, ch, x_disp, tx, vy, n, vmin) in enumerate(loaded):
        color = SLOT_COLORS[i % len(SLOT_COLORS)]
        ax = axes[i]
        ax.set_facecolor(SURFACE)

        # Agg's alpha compositing renders nothing below ~0.004-0.005 (empirically
        # verified) regardless of how many overlapping strokes accumulate --
        # so the floor here is a hard rendering constraint, not a style choice.
        alpha = float(np.clip(TARGET_ALPHA_COVERAGE / max(n, 1), 0.006, 0.05))
        ax.plot(tx, vy, color=color, alpha=alpha, linewidth=0.6, zorder=2)

        ax.axvspan(*SIGNAL_T_RANGE, color=color, alpha=0.10, zorder=1,
                   label=f"signal window {SIGNAL_T_RANGE} ns")
        ax.axhline(THRESHOLD_MV, color=MUTED, linewidth=1, linestyle="--", zorder=3)
        ax.set_ylim(y_lo, y_hi)  # shared across all panels in this figure

        ax.set_title(f"{label} (ch{ch}), X={x_disp} mm  --  n={n}", color=INK, fontsize=12)
        ax.set_xlabel("Time (ns)", color=INK, fontsize=10)
        ax.set_ylabel("Voltage (mV)", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=SECONDARY, labelsize=9)
        legend = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        for t in legend.get_texts():
            t.set_color(INK)

    fig.suptitle(suptitle, color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[overlay] wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 260812: peak position, P14/P10/P6 ---
    cache_260812 = os.path.join(CACHE_ROOT, "260812")
    plot_overlay(
        cache_260812,
        [("P14", "AA0812X17", 2, 17), ("P10", "AA0812X23", 3, 40), ("P6", "AA0812X67", 4, 67)],
        "260812: raw waveforms at each pad's peak position (all shots overlaid)",
        os.path.join(OUT_DIR, "waveforms_260812_peak.png"),
    )

    # --- 260814: boundary, P14/P10/P6 (same pad set as 260812) @ X=28mm ---
    cache_260814 = os.path.join(CACHE_ROOT, "260814")
    plot_overlay(
        cache_260814,
        [("P14", "AA0814X28", 2, 28), ("P10", "AA0814X28", 3, 28), ("P6", "AA0814X28", 4, 28)],
        "260814: raw waveforms at the P14/P10 boundary (all shots overlaid)",
        os.path.join(OUT_DIR, "waveforms_260814_boundary_P14P10.png"),
    )

    # --- 260814XY: peak position, P13/P9/P5 ---
    cache_260814xy = os.path.join(CACHE_ROOT, "260814XY")
    plot_overlay(
        cache_260814xy,
        [("P13", "AA0814X17Y0", 2, 17), ("P9", "AA0814X36Y0", 3, 36), ("P5", "AA0814X68Y0", 4, 68)],
        "260814XY: raw waveforms at each pad's peak position (all shots overlaid)",
        os.path.join(OUT_DIR, "waveforms_260814XY_peak.png"),
    )

    # --- 260814XY: boundaries, P13/P9/P5 @ X=28 (P13/P9) and X=53 (P9/P5) ---
    plot_overlay(
        cache_260814xy,
        [("P13", "AA0814X28Y0", 2, 28), ("P9", "AA0814X28Y0", 3, 28), ("P5", "AA0814X28Y0", 4, 28)],
        "260814XY: raw waveforms at the P13/P9 boundary (all shots overlaid)",
        os.path.join(OUT_DIR, "waveforms_260814XY_boundary_P13P9.png"),
    )
    plot_overlay(
        cache_260814xy,
        [("P13", "AA0814X53Y0", 2, 53), ("P9", "AA0814X53Y0", 3, 53), ("P5", "AA0814X53Y0", 4, 53)],
        "260814XY: raw waveforms at the P9/P5 boundary (all shots overlaid)",
        os.path.join(OUT_DIR, "waveforms_260814XY_boundary_P9P5.png"),
    )


if __name__ == "__main__":
    main()
