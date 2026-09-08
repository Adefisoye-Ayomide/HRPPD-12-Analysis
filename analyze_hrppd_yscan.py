"""
analyze_hrppd_yscan.py
========================
260901XY is a Y-scan (X fixed at 0, Y varies 0-86mm), not an X-scan like
every prior "XY" run -- same per-event pulse-height methodology as
analyze_hrppd.py (baseline-subtract, min in the signal window, OR-gate
event selection across channels, rail/clip detection), just iterated over Y
instead of X.

No pad mapping was given for this run, so channels are labeled ch2/ch3/ch4
directly rather than guessed pad names -- relabel once the mapping is known.

Usage:
    python3 analyze_hrppd_yscan.py --run 260901XY
"""

import os, glob, csv, argparse
from collections import Counter
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE   = (40.0, 65.0)
CONTROL_T_RANGE  = (10.0, 35.0)
THRESHOLD_MV     = -25.0
CLIP_EPS_MV      = 0.5
RAIL_MIN_REPEATS = 5
CHANNELS         = [2, 3, 4]

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def detect_rail(vals_mv, eps=CLIP_EPS_MV, min_repeats=RAIL_MIN_REPEATS):
    if len(vals_mv) == 0:
        return None, np.zeros(0, dtype=bool)
    binned = np.round(vals_mv / eps) * eps
    counts = Counter(binned)
    rail_val, rail_count = min(counts.items(), key=lambda kv: kv[0]) if False else (None, 0)
    best_val, best_count = None, 0
    for val, cnt in counts.items():
        if cnt >= min_repeats and cnt > best_count:
            best_val, best_count = val, cnt
    if best_val is None:
        return None, np.zeros(len(vals_mv), dtype=bool)
    mask = np.abs(binned - best_val) < eps / 2
    return float(best_val), mask


def find_positions(cache_dir):
    files = glob.glob(os.path.join(cache_dir, "*_ch2_*.npz"))
    ys = set()
    for f in files:
        base = os.path.basename(f)
        tag = base.split("_ch2_")[0]  # e.g. AA0901X0Y13
        y = int(tag.split("Y")[1])
        ys.add(y)
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


def process_position(args):
    cache_dir, y = args
    per_ch, per_ch_ctrl = {}, {}
    for ch in CHANNELS:
        per_ch[ch], per_ch_ctrl[ch] = load_channel_mins(cache_dir, y, ch)

    n = min(len(per_ch[ch]) for ch in CHANNELS)
    if n == 0:
        return {"y": y, "n_total": 0, "n_selected": 0,
                "mean": {c: np.nan for c in CHANNELS}, "sem": {c: np.nan for c in CHANNELS},
                "pass_rate": {c: np.nan for c in CHANNELS},
                "n_clipped": {c: 0 for c in CHANNELS}, "clip_value": {c: np.nan for c in CHANNELS}}

    mat = np.stack([per_ch[c][:n] for c in CHANNELS], axis=1)
    ctrl_mat = np.stack([per_ch_ctrl[c][:n] for c in CHANNELS], axis=1)
    passed = mat <= THRESHOLD_MV
    selected = passed.any(axis=1)  # OR across all 3 channels

    result = {"y": y, "n_total": n, "n_selected": int(selected.sum()),
              "mean": {}, "sem": {}, "pass_rate": {}, "n_clipped": {}, "clip_value": {}}

    for j, ch in enumerate(CHANNELS):
        own_pass_idx = np.where(mat[:, j] <= THRESHOLD_MV)[0]
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

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    args = ap.parse_args()

    cache_dir = os.path.join(args.cache_root, args.run)
    ys = find_positions(cache_dir)
    print(f"[yscan] {len(ys)} Y positions found in {cache_dir}")

    tasks = [(cache_dir, y) for y in ys]
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(process_position, tasks)
    results.sort(key=lambda r: r["y"])

    out_csv = f"hrppd_yscan_results_{args.run}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y_mm", "channel", "mean_pulse_height_mV", "sem_mV", "pass_rate",
                    "n_selected", "n_total", "n_clipped", "clip_value_mV"])
        for r in results:
            for ch in CHANNELS:
                w.writerow([r["y"], ch, r["mean"][ch], r["sem"][ch], r["pass_rate"][ch],
                            r["n_selected"], r["n_total"], r["n_clipped"][ch], r["clip_value"][ch]])
    print(f"[yscan] wrote {out_csv}")

    for ch in CHANNELS:
        total_clipped = sum(r["n_clipped"][ch] for r in results)
        if total_clipped:
            worst = max(results, key=lambda r: r["n_clipped"][ch])
            print(f"[yscan]   [CLIP] ch{ch}: {total_clipped} events excluded as clipped/railed "
                  f"(worst: y={worst['y']}mm, {worst['n_clipped'][ch]} events pinned at {worst['clip_value'][ch]:.1f} mV)")

    # --- plot: mean pulse height vs Y position ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, ch in enumerate(CHANNELS):
        means = [r["mean"][ch] for r in results]
        sems = [r["sem"][ch] for r in results]
        yy = [r["y"] for r in results]
        ax.errorbar(yy, means, yerr=sems, color=SLOT_COLORS[i], linewidth=2,
                    marker="o", markersize=4, capsize=2, elinewidth=1, label=f"ch{ch}", zorder=3)
    ax.set_xlabel("Y position (mm)", color=INK, fontsize=11)
    ax.set_ylabel("Mean pulse height (mV)", color=INK, fontsize=11)
    ax.set_title(f"{args.run}: mean pulse height vs Y position (X fixed at 12.5mm)\n"
                 f"(events selected where any channel crosses {THRESHOLD_MV:g} mV; pad mapping not yet specified)",
                 color=INK, fontsize=12, pad=12)
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
    out_png = f"hrppd_yscan_{args.run}_pulse_height.png"
    fig.savefig(out_png, facecolor=SURFACE)
    print(f"[yscan] wrote {out_png}")


if __name__ == "__main__":
    main()
