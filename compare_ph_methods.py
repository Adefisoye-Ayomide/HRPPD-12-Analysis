"""
compare_ph_methods.py
======================
Compares three ways of turning one event's raw waveform into a "pulse
height" number, run on the full 260814XY X-scan:

  A. min-search   -- the existing method: -(min(v)) in the signal window.
                      Decided by a single sample; biased upward by noise
                      (an extreme-value/order-statistic effect).
  B. charge        -- integrate the baseline-subtracted trace over the
                      signal window (sum(v)*dt / 50 ohm -> pC). Random noise
                      averages toward zero over many samples instead of
                      being hunted for, so it should be far less biased.
  C. matched filter -- project the event onto a per-pad TEMPLATE (the mean
                      pulse shape, built from many events at that pad's own
                      peak, normalized to unit depth) via a linear
                      least-squares amplitude fit. Uses every sample in the
                      window, weighted by the template shape, so no single
                      noisy sample dominates. Reported in mV, same scale as
                      method A, for direct comparison. Does NOT search over
                      timing offsets -- a fixed template, so it tolerates
                      (but does not fully correct for) event-to-event timing
                      jitter (~380-400 ps, from the TTS work -- small
                      compared to the ~10 ns template width).

All three are computed on the SAME OR-gated event population (gate decided
by method A crossing -25 mV, exactly as analyze_hrppd.py does) so the
comparison isolates "does the estimator change the answer" from "does it
change which events count."

Outputs:
    hrppd_compare_methods_260814XY.csv          -- mean per (x, pad, method)
    hrppd_compare_methods_260814XY_curves.png   -- mean-vs-position, all 3
    hrppd_compare_methods_260814XY_fits.png     -- template + example fits
"""

import os, glob, csv
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
RUN = "260814XY"

CHANNEL_PAD = {2: "P13", 3: "P9", 4: "P5"}
PAD_ORDER = list(CHANNEL_PAD.values())
PEAK_X = {"P13": 17, "P9": 36, "P5": 68}  # rough peaks, for template construction only

BASELINE_T_RANGE = (-10.0, 0.0)
SIGNAL_T_RANGE = (40.0, 65.0)
THRESHOLD_MV = -25.0
IMPEDANCE_OHM = 50.0

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

_TEMPLATE = {}  # pad -> (t_window, template_normalized), filled before Pool workers spawn


def build_template(cache_dir, pad, ch, x, n_sample=3000):
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
    depth = -mean_trace.min()
    template = mean_trace / depth  # normalized so template.min() == -1
    return t_ref, template


def find_positions(cache_dir, any_channel):
    import re
    FNAME_RE = re.compile(r"X(\d+)Y0_ch(\d+)_(\d+)\.npz$")
    xs = set()
    for f in glob.glob(os.path.join(cache_dir, f"*Y0_ch{any_channel}_*.npz")):
        m = FNAME_RE.search(f)
        if m:
            xs.add(int(m.group(1)))
    return sorted(xs)


def per_event_all_methods(t, v, pad):
    b_lo, b_hi = BASELINE_T_RANGE
    s_lo, s_hi = SIGNAL_T_RANGE
    b = np.median(v[(t >= b_lo) & (t <= b_hi)])
    m = (t >= s_lo) & (t <= s_hi)
    w = v[m] - b  # baseline-subtracted, negative-going

    ph_min = -w.min()

    dt_ns = float(np.median(np.diff(t)))
    charge_pC = -np.sum(w) * dt_ns / IMPEDANCE_OHM

    template = _TEMPLATE[pad]
    n = min(len(template), len(w))  # guard against +/-1 sample edge-of-window rounding
    tmpl_n, w_n = template[:n], w[:n]
    # template.min() == -1 by construction, so fitting w ~= A*template via least squares
    # (A = dot(template,w)/dot(template,template)) already comes out as a POSITIVE mV
    # depth -- no extra sign flip needed (template and w are both negative-going, so
    # their dot product is already positive).
    amp_matched = float(np.dot(tmpl_n, w_n) / np.dot(tmpl_n, tmpl_n))

    return ph_min, charge_pC, amp_matched


def process_position(args):
    cache_dir, x = args
    per_ch = {}
    for ch, pad in CHANNEL_PAD.items():
        files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))
        per_ch[pad] = files

    n = min(len(per_ch[p]) for p in PAD_ORDER)
    if n == 0:
        return None

    ph_min = {p: np.empty(n) for p in PAD_ORDER}
    charge = {p: np.empty(n) for p in PAD_ORDER}
    matched = {p: np.empty(n) for p in PAD_ORDER}

    for p in PAD_ORDER:
        for i in range(n):
            with np.load(per_ch[p][i]) as z:
                t, v = z["t"], z["v"]
            a, b, c = per_event_all_methods(t, v, p)
            ph_min[p][i], charge[p][i], matched[p][i] = a, b, c

    mat = np.stack([ph_min[p] for p in PAD_ORDER], axis=1)  # positive-magnitude mV
    selected = (mat >= -THRESHOLD_MV).any(axis=1)  # -THRESHOLD_MV = +25.0

    out = {"x": x, "n_total": n, "n_selected": int(selected.sum())}
    for p in PAD_ORDER:
        if selected.sum() > 0:
            out[f"{p}_min"] = float(ph_min[p][selected].mean())
            out[f"{p}_charge"] = float(charge[p][selected].mean())
            out[f"{p}_matched"] = float(matched[p][selected].mean())
        else:
            out[f"{p}_min"] = out[f"{p}_charge"] = out[f"{p}_matched"] = float("nan")
    return out


def _init_worker(templates):
    global _TEMPLATE
    _TEMPLATE = templates


def main():
    cache_dir = os.path.join(CACHE_ROOT, RUN)

    print("[compare] building per-pad templates...")
    templates = {}
    t_windows = {}
    for pad, ch in [(p, ch) for ch, p in CHANNEL_PAD.items()]:
        t_win, tmpl = build_template(cache_dir, pad, ch, PEAK_X[pad])
        templates[pad] = tmpl
        t_windows[pad] = t_win
        print(f"[compare]   {pad}: template built from X={PEAK_X[pad]}mm, {len(tmpl)} samples")

    xs = find_positions(cache_dir, 2)
    print(f"[compare] {len(xs)} X positions found")

    tasks = [(cache_dir, x) for x in xs]
    with Pool(processes=max(1, cpu_count() - 1), initializer=_init_worker, initargs=(templates,)) as pool:
        results = [r for r in pool.map(process_position, tasks) if r is not None]
    results.sort(key=lambda r: r["x"])

    out_csv = f"hrppd_compare_methods_{RUN}.csv"
    with open(out_csv, "w", newline="") as fh:
        fieldnames = ["x_mm", "n_total", "n_selected"] + \
                     [f"{p}_{m}" for p in PAD_ORDER for m in ("min", "charge", "matched")]
        w = csv.writer(fh)
        w.writerow(fieldnames)
        for r in results:
            row = [r["x"], r["n_total"], r["n_selected"]]
            for p in PAD_ORDER:
                row += [r[f"{p}_min"], r[f"{p}_charge"], r[f"{p}_matched"]]
            w.writerow(row)
    print(f"[compare] wrote {out_csv}")

    # ---------------------------------------------------------------- plot curves
    xs_arr = np.array([r["x"] for r in results])
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for j, pad in enumerate(PAD_ORDER):
        color = SLOT_COLORS[j % len(SLOT_COLORS)]
        ax_top, ax_bot = axes[0, j], axes[1, j]
        for ax in (ax_top, ax_bot):
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(MUTED)
            ax.tick_params(colors=SECONDARY, labelsize=8.5)

        min_vals = np.array([r[f"{pad}_min"] for r in results])
        matched_vals = np.array([r[f"{pad}_matched"] for r in results])
        charge_vals = np.array([r[f"{pad}_charge"] for r in results])

        ax_top.plot(xs_arr, min_vals, color=color, linewidth=1.8, marker="o", markersize=2.5,
                    label="min-search", zorder=3)
        ax_top.plot(xs_arr, matched_vals, color=INK, linewidth=1.4, linestyle="--", marker="s",
                    markersize=2, label="matched filter", zorder=4, alpha=0.85)
        ax_top.set_title(f"{pad}: min-search vs matched filter (mV)", color=INK, fontsize=10.5)
        ax_top.set_xlabel("X (mm)", color=INK, fontsize=9)
        ax_top.set_ylabel("mean pulse height (mV)", color=INK, fontsize=9)
        leg = ax_top.legend(frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(INK)

        ax_bot.plot(xs_arr, charge_vals, color=color, linewidth=1.8, marker="o", markersize=2.5, zorder=3)
        ax_bot.set_title(f"{pad}: charge (integral)", color=INK, fontsize=10.5)
        ax_bot.set_xlabel("X (mm)", color=INK, fontsize=9)
        ax_bot.set_ylabel("mean charge (pC)", color=INK, fontsize=9)

    fig.suptitle(f"{RUN}: mean pulse height vs position -- 3 estimators, same gated events",
                 color=INK, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = f"hrppd_compare_methods_{RUN}_curves.png"
    fig.savefig(out_png, facecolor=SURFACE)
    print(f"[compare] wrote {out_png}")

    # ---------------------------------------------------------------- sample fits
    fig2, axes2 = plt.subplots(2, 3, figsize=(17, 8.5), dpi=150)
    fig2.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(7)

    for j, pad in enumerate(PAD_ORDER):
        ch = [c for c, p in CHANNEL_PAD.items() if p == pad][0]
        x = PEAK_X[pad]
        files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))
        t_win, tmpl = t_windows[pad], templates[pad]
        color = SLOT_COLORS[j % len(SLOT_COLORS)]

        # find two examples: a strong one and a middling one, among events that pass threshold
        cand_idx = rng.choice(len(files), size=min(400, len(files)), replace=False)
        cand = []
        for i in cand_idx:
            with np.load(files[i]) as z:
                t, v = z["t"], z["v"]
            b = np.median(v[(t >= BASELINE_T_RANGE[0]) & (t <= BASELINE_T_RANGE[1])])
            m = (t >= SIGNAL_T_RANGE[0]) & (t <= SIGNAL_T_RANGE[1])
            w = v[m] - b
            if -w.min() >= -THRESHOLD_MV:
                cand.append((i, -w.min(), t, v, b, w))
        cand.sort(key=lambda c: c[1])
        picks = [cand[int(len(cand) * 0.85)], cand[int(len(cand) * 0.4)]] if len(cand) > 5 else cand[:2]

        for row, (i, ph, t, v, b, w) in enumerate(picks):
            ax = axes2[row, j]
            ax.set_facecolor(SURFACE)
            m_full = (t >= 30) & (t <= 75)
            ax.plot(t[m_full], (v - b)[m_full], color=color, linewidth=1.4, label="raw trace", zorder=3)

            n_fit = min(len(tmpl), len(w))
            amp = float(np.dot(tmpl[:n_fit], w[:n_fit]) / np.dot(tmpl[:n_fit], tmpl[:n_fit]))
            ax.plot(t_win[:n_fit], amp * tmpl[:n_fit], color=INK, linewidth=1.6, linestyle="--",
                    label=f"template fit (A={amp:.1f}mV)", zorder=4)

            charge_pC = -np.sum(w) * float(np.median(np.diff(t))) / IMPEDANCE_OHM
            ax.axhline(-ph, color=MUTED, linewidth=1, linestyle=":", zorder=2,
                       label=f"min-search = {ph:.1f}mV")

            ax.set_title(f"{pad} (ch{ch}) example event -- min={ph:.1f}mV, "
                         f"matched={amp:.1f}mV, charge={charge_pC:.2f}pC",
                         color=INK, fontsize=9.5)
            ax.set_xlabel("Time (ns)", color=INK, fontsize=9)
            ax.set_ylabel("mV", color=INK, fontsize=9)
            ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(MUTED)
            ax.tick_params(colors=SECONDARY, labelsize=8)
            leg = ax.legend(frameon=False, fontsize=7.5, loc="lower right")
            for t2 in leg.get_texts():
                t2.set_color(INK)

    fig2.suptitle(f"{RUN}: template fit vs raw trace, 2 example events per pad (at each pad's peak)",
                  color=INK, fontsize=14)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    out_png2 = f"hrppd_compare_methods_{RUN}_fits.png"
    fig2.savefig(out_png2, facecolor=SURFACE)
    print(f"[compare] wrote {out_png2}")


if __name__ == "__main__":
    main()
