"""
analyze_260902XY_yscan.py
============================
260902XY is a Y-position scan at fixed X=0 (Y=0..86) -- unlike every prior
"XY"-named run, which was actually a fixed-Y=0 X-scan. Filenames are
AA0901X0Y{y}_ch{N}_*.npz.

Pad identity for ch2/ch3/ch4 was not specified for this run, so channels are
labeled generically (ch2/ch3/ch4) rather than guessing pad names. Same
methodology as analyze_hrppd.py otherwise: baseline-subtract (median,
pre-pulse window), per-event min in the signal window, OR-gate event
selection across all 3 channels at THRESHOLD_MV, per-channel rail/clip
detection and exclusion.

Usage:
    python3 analyze_260902XY_yscan.py
"""

import os, glob, re
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
RUN = "260902XY"

CHANNELS = [2, 3, 4]
CHANNEL_PAD = {2: "P9", 3: "P10", 4: "P11"}
BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)
CONTROL_T_RANGE = (10.0, 35.0)
THRESHOLD_MV = -25.0
CLIP_EPS_MV = 0.5
RAIL_MIN_REPEATS = 5

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

FNAME_RE = re.compile(r"X0Y(\d+)_ch")


def find_positions(cache_dir):
    ys = set()
    for f in glob.glob(os.path.join(cache_dir, f"*X0Y*_ch{CHANNELS[0]}_*.npz")):
        m = FNAME_RE.search(f)
        if m:
            ys.add(int(m.group(1)))
    return sorted(ys)


def load_channel_mins(cache_dir, y, ch):
    files = sorted(glob.glob(os.path.join(cache_dir, f"*X0Y{y}_ch{ch}_*.npz")))
    mins = np.empty(len(files), dtype=np.float64)
    ctrl_mins = np.empty(len(files), dtype=np.float64)
    b_lo, b_hi = BASELINE_T_RANGE
    s_lo, s_hi = SIGNAL_T_RANGE
    c_lo, c_hi = CONTROL_T_RANGE
    for i, f in enumerate(files):
        with np.load(f) as z:
            t, v = z["t"], z["v"]
        baseline = np.median(v[(t >= b_lo) & (t <= b_hi)])
        sig_mask = (t >= s_lo) & (t <= s_hi)
        ctrl_mask = (t >= c_lo) & (t <= c_hi)
        mins[i] = (v[sig_mask] - baseline).min()
        ctrl_mins[i] = (v[ctrl_mask] - baseline).min()
    return mins, ctrl_mins


def detect_rail(vals_mv, eps=CLIP_EPS_MV, min_repeats=RAIL_MIN_REPEATS):
    mask = np.zeros(len(vals_mv), dtype=bool)
    if len(vals_mv) == 0:
        return None, mask
    binned = np.round(vals_mv / eps) * eps
    counts = Counter(binned.tolist())
    val, n = counts.most_common(1)[0]
    if n < min_repeats:
        return None, mask
    return float(val), np.abs(binned - val) < 1e-9


def process_position(args):
    cache_dir, y, threshold = args
    per_ch, per_ch_ctrl = {}, {}
    for ch in CHANNELS:
        per_ch[ch], per_ch_ctrl[ch] = load_channel_mins(cache_dir, y, ch)

    n = min(len(per_ch[ch]) for ch in CHANNELS)
    result = {"y": y, "n_total": n, "n_selected": 0, "mean": {}, "sem": {},
              "pass_rate": {}, "ctrl_mean": {}, "ctrl_median": {}, "n_clipped": {}, "clip_value": {}}
    if n == 0:
        for ch in CHANNELS:
            result["mean"][ch] = np.nan
            result["sem"][ch] = np.nan
            result["pass_rate"][ch] = np.nan
            result["ctrl_mean"][ch] = np.nan
            result["ctrl_median"][ch] = np.nan
            result["n_clipped"][ch] = 0
            result["clip_value"][ch] = np.nan
        return result

    mat = np.stack([per_ch[ch][:n] for ch in CHANNELS], axis=1)
    ctrl_mat = np.stack([per_ch_ctrl[ch][:n] for ch in CHANNELS], axis=1)
    passed = mat <= threshold
    selected = passed.any(axis=1)  # OR-gate across all 3 channels
    result["n_selected"] = int(selected.sum())

    for j, ch in enumerate(CHANNELS):
        own_pass_idx = np.where(mat[:, j] <= threshold)[0]
        rail_val, rail_mask_own = detect_rail(mat[own_pass_idx, j]) if own_pass_idx.size else (None, np.array([], bool))
        clipped_global = np.zeros(n, dtype=bool)
        if rail_val is not None:
            clipped_global[own_pass_idx[rail_mask_own]] = True
        result["n_clipped"][ch] = int(clipped_global.sum())
        result["clip_value"][ch] = rail_val if rail_val is not None else float("nan")

        sel_not_clipped = selected & (~clipped_global)
        vals = -mat[sel_not_clipped, j]
        if vals.size > 0:
            result["mean"][ch] = float(np.mean(vals))
            result["sem"][ch] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            result["pass_rate"][ch] = float(passed[sel_not_clipped, j].mean())
        else:
            result["mean"][ch] = np.nan
            result["sem"][ch] = np.nan
            result["pass_rate"][ch] = np.nan

        ctrl_vals = -ctrl_mat[:, j]
        result["ctrl_mean"][ch] = float(np.mean(ctrl_vals))
        result["ctrl_median"][ch] = float(np.median(ctrl_vals))

    return result


def main():
    cache_dir = os.path.join(CACHE_ROOT, RUN)
    ys = find_positions(cache_dir)
    print(f"[yscan] {len(ys)} Y positions found in {cache_dir}: {ys[0]}..{ys[-1]}")

    tasks = [(cache_dir, y, THRESHOLD_MV) for y in ys]
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(process_position, tasks)
    results.sort(key=lambda r: r["y"])

    # --- CSV ---
    import csv
    out_csv = f"hrppd_yscan_results_{RUN}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y_mm", "pad", "channel", "mean_pulse_height_mV", "sem_mV", "pass_rate",
                    "n_selected", "n_total", "ctrl_mean_mV", "ctrl_median_mV", "n_clipped", "clip_value_mV"])
        for r in results:
            for ch in CHANNELS:
                w.writerow([r["y"], CHANNEL_PAD[ch], ch, r["mean"][ch], r["sem"][ch], r["pass_rate"][ch],
                            r["n_selected"], r["n_total"], r["ctrl_mean"][ch], r["ctrl_median"][ch],
                            r["n_clipped"][ch], r["clip_value"][ch]])
    print(f"[yscan] wrote {out_csv}")

    for ch in CHANNELS:
        total_clipped = sum(r["n_clipped"][ch] for r in results)
        n_pos_clipped = sum(1 for r in results if r["n_clipped"][ch] > 0)
        if total_clipped:
            worst = max(results, key=lambda r: r["n_clipped"][ch])
            print(f"[yscan] [CLIP] {CHANNEL_PAD[ch]} (ch{ch}): {total_clipped} events excluded as clipped/railed across "
                  f"{n_pos_clipped} positions (worst: y={worst['y']}mm, {worst['n_clipped'][ch]} events "
                  f"pinned at {worst['clip_value'][ch]:.1f} mV)")

    # --- plot: mean pulse height vs Y position ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, ch in enumerate(CHANNELS):
        means = [r["mean"][ch] for r in results]
        sems = [r["sem"][ch] for r in results]
        ax.errorbar(ys, means, yerr=sems, color=SLOT_COLORS[i], linewidth=2,
                    marker="o", markersize=4, capsize=2, elinewidth=1, label=CHANNEL_PAD[ch], zorder=3)
    ax.set_xlabel("Y position (mm)", color=INK, fontsize=11)
    ax.set_ylabel("Mean pulse height (mV)", color=INK, fontsize=11)
    ax.set_title("HRPPD Y scan mean pulse height per channel",
                 color=INK, fontsize=13, pad=12)
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
    out_png = f"hrppd_yscan_{RUN}_pulse_height.png"
    fig.savefig(out_png, facecolor=SURFACE)
    print(f"[yscan] wrote {out_png}")


if __name__ == "__main__":
    main()
