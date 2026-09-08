"""
analyze_hrppd_timing.py
========================
Transit-time-spread (TTS) / timing-resolution analysis from the same NPZ
cache used by analyze_hrppd.py.

METHODOLOGY (revised to follow Shin et al., "Advances in the Large Area
Picosecond Photo-Detector (LAPPD): 8x8 MCP-PMT with Capacitively Coupled
Readout", arXiv:2212.03208, Sec 3.3/5.3):

For each pad, at the X position where it shows its peak response (read from
a results CSV produced by analyze_hrppd.py), this pairs every event's pad
waveform with the laser-sync reference on ch1 (captured on the same
trigger, splitting off the same laser pulse the pad sees) and measures the
time between them.

  1. Baseline-subtract both traces (median over each channel's own
     pre-pulse window).
  2. Per-event timing uses the pulse CENTROID by default (Shin et al. Sec
     3.3: "the weighted mean of the corresponding waveform"), not a
     constant-fraction discriminator: t_centroid = sum(t_i * w_i) /
     sum(w_i), with w_i = max(0, -v_i) over the signal window (only the
     negative-going pulse contributes weight, so baseline noise fluctuating
     positive doesn't bias the sum). The centroid integrates the whole
     pulse shape rather than reading a single threshold-crossing sample, so
     it averages down sample noise better for a well-isolated pulse. CFD
     (50%-of-peak, time-walk-cancelling) is still available via
     --timing-method cfd for comparison against the prior version of this
     analysis.
  3. dt = t_pad_centroid - t_ref_centroid.
  4. The dt distribution is fit as a Gaussian CORE plus an ex-Gaussian
     (exponentially-modified Gaussian) TAIL, matching Shin et al. Fig 9/Eq
     3.6: fit the core on an iteratively-trimmed window around the peak
     (so the tail doesn't bias sigma), then fit an ex-Gaussian to the
     residual (data minus core fit) on the late side. sigma_core -- not
     the naive whole-distribution std/single-Gaussian sigma -- is the
     reported TTS. The tail is physically attributed (Shin et al.) to
     backscattered photoelectrons or secondary electrons off the entry
     MCP's top interstitial layer, producing late-arriving pulses; it is
     not part of the detector's core timing resolution.
  5. Optional --laser-sigma-ps applies Shin et al. Eq 3.6: sigma_core_true
     = sqrt(sigma_core^2 - sigma_LW^2), removing the finite laser pulse
     width's own contribution to the measured spread. Off by default (we
     don't have an independently-measured sigma_LW for our own laser on
     file) -- when omitted, the reported sigma_core is the RAW measured
     value and is explicitly labeled as not laser-width-corrected, same
     caveat Shin et al. give for their own uncorrected numbers.

Event selection here is deliberately NOT the OR-gate used for pulse-height
averaging in analyze_hrppd.py: a pad's timing can only be measured on
events where THAT pad itself has a real pulse, so each pad is self-gated
(its own signal-window peak found) here, evaluated only at its peak
position where its self-trigger pass rate is highest.

The mean of the dt distribution reflects an arbitrary, run-fixed cable/
scope-channel delay offset and is not itself physically meaningful. The
WIDTH is the transit-time spread.

Minimum-depth floor (--min-depth-mv, default -25 mV): a marginal pulse
that just clears the spatial-gate pass threshold is indistinguishable from
ordinary baseline noise wandering. Requiring a substantially deeper peak
before an event counts for timing avoids spurious early "hits" from noise.

Usage:
    python analyze_hrppd_timing.py --run 260807XY \\
        --results-csv hrppd_xscan_results_260807XY.csv \\
        --channel-map "2:P12,3:P8,4:P4" --ref-channel 1

    # with the Shin et al. laser-width correction, if you have it measured:
    python analyze_hrppd_timing.py --run 260807XY \\
        --results-csv hrppd_xscan_results_260807XY.csv \\
        --channel-map "2:P12,3:P8,4:P4" --ref-channel 1 --laser-sigma-ps 27
"""

import os, glob, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import exponnorm

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"

PAD_BASELINE_T_RANGE = (-10.0, 0.0)
PAD_SIGNAL_T_RANGE   = (40.0, 65.0)
REF_BASELINE_T_RANGE = (-30.0, -5.0)
REF_SEARCH_T_RANGE   = (-5.0, 15.0)
THRESHOLD_MV = -5.0
TIMING_MIN_DEPTH_MV = -25.0
CFD_FRAC = 0.5
CORE_TRIM_SIGMA = 2.5  # +/- this many sigma kept when iteratively isolating the core from the tail

SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def cfd_time(t, v, lo, hi, frac, threshold_mv=None):
    """
    v must already be baseline-subtracted. Finds the minimum within [lo,hi],
    then the first (earliest) linearly-interpolated crossing of
    frac*min on the leading edge. Returns (t_cross, min_v) or (None, min_v)
    if no real pulse / no crossing found.
    """
    mask = (t >= lo) & (t <= hi)
    tt, vv = t[mask], v[mask]
    if len(tt) == 0:
        return None, None
    imin = int(np.argmin(vv))
    peak = vv[imin]
    if threshold_mv is not None and peak > threshold_mv:
        return None, peak
    thresh = frac * peak  # peak is negative; thresh is less deep than peak
    sub_t, sub_v = tt[:imin + 1], vv[:imin + 1]
    below = np.where(sub_v <= thresh)[0]
    if len(below) == 0:
        return None, peak
    i = int(below[0])
    if i == 0:
        tc = sub_t[0]
    else:
        t0, t1, v0, v1 = sub_t[i - 1], sub_t[i], sub_v[i - 1], sub_v[i]
        tc = t0 if v1 == v0 else t0 + (thresh - v0) * (t1 - t0) / (v1 - v0)
    return tc, peak


def centroid_time(t, v, lo, hi, threshold_mv=None):
    """
    v must already be baseline-subtracted. Pulse-centroid timing (Shin et
    al. Sec 3.3): the charge-weighted mean time of the negative-going pulse
    within [lo, hi]. Weight is clip(-v, 0, None) so baseline noise
    fluctuating to the positive side contributes zero weight rather than
    biasing the mean. Returns (t_centroid, min_v) or (None, min_v) if no
    real pulse / no weight.
    """
    mask = (t >= lo) & (t <= hi)
    tt, vv = t[mask], v[mask]
    if len(tt) == 0:
        return None, None
    peak = vv.min()
    if threshold_mv is not None and peak > threshold_mv:
        return None, peak
    w = np.clip(-vv, 0.0, None)
    total_w = w.sum()
    if total_w <= 0:
        return None, peak
    tc = float(np.sum(tt * w) / total_w)
    return tc, peak


def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def exgauss(x, a, mu, sigma, tau):
    """Ex-Gaussian (exponentially-modified Gaussian) tail, amplitude-scaled.
    exponnorm.pdf is already area-1, so `a` directly free-scales to counts."""
    tau = max(abs(tau), 1e-6)
    sigma = max(abs(sigma), 1e-6)
    k = tau / sigma
    return a * exponnorm.pdf(x, k, loc=mu, scale=sigma)


def fit_core_and_tail(dt_ns, n_bins=60):
    """
    Sequential core+tail fit matching Shin et al. Fig 9: (1) Gaussian fit
    to the core, isolated by iteratively trimming to +/-CORE_TRIM_SIGMA
    around the peak so the tail doesn't bias sigma_core; (2) ex-Gaussian
    fit to the residual (data - core) on the late side.

    Returns a dict with core/tail fit params and both histograms for
    plotting, or None fields where a fit wasn't possible.
    """
    counts, edges = np.histogram(dt_ns, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]  # kept FIXED through the trim iterations below (see note)

    # --- Step 1: iteratively-trimmed Gaussian core fit ---
    # Seed robustly: mode (peak bin) + MAD-based sigma, NOT median/std --
    # with a heavy tail (routinely ~20-30% of events here), raw std is
    # inflated by the tail and can send the very first trim window off
    # into the tail, from which the iteration never recovers.
    mu = float(centers[np.argmax(counts)])
    sigma = float(1.4826 * np.median(np.abs(dt_ns - mu))) or float(np.std(dt_ns))
    core_popt, core_ok = None, False
    window = dt_ns[(dt_ns >= mu - CORE_TRIM_SIGMA * sigma) & (dt_ns <= mu + CORE_TRIM_SIGMA * sigma)]
    for _ in range(4):
        if len(window) < 20:
            break
        # Bin width held fixed (not bin COUNT) across iterations: refitting a
        # shrinking window with a fixed *number* of bins shrinks the bin
        # width each pass, which silently rescales the fitted amplitude and
        # makes it incomparable to the full-range histogram used for
        # plotting/residual below (amplitude ~ counts-per-bin, bin-width
        # dependent).
        n_local = max(10, int(round((window.max() - window.min()) / bin_width)))
        c, e = np.histogram(window, bins=n_local)
        ctr = 0.5 * (e[:-1] + e[1:])
        p0 = [max(c.max(), 1), mu, max(sigma, 1e-3)]
        try:
            popt, _ = curve_fit(gauss, ctr, c, p0=p0, maxfev=5000)
            mu, sigma = float(popt[1]), abs(float(popt[2]))
            core_popt, core_ok = popt, True
        except Exception:
            break
        window = dt_ns[(dt_ns >= mu - CORE_TRIM_SIGMA * sigma) & (dt_ns <= mu + CORE_TRIM_SIGMA * sigma)]

    if not core_ok:
        mu, sigma = float(np.median(dt_ns)), float(np.std(dt_ns))
        core_popt = [float(len(dt_ns)), mu, sigma]

    # --- Step 2: ex-Gaussian fit to the residual (late-side excess) ---
    core_pred = gauss(centers, *core_popt)
    residual = counts - core_pred
    tail_mask = (centers > mu) & (residual > 0)
    tail_popt, tail_ok = None, False
    if tail_mask.sum() >= 6 and residual[tail_mask].sum() > 0:
        try:
            a0 = residual[tail_mask].max()
            tau0 = max(dt_ns.max() - mu, sigma)
            popt, _ = curve_fit(
                exgauss, centers[tail_mask], residual[tail_mask],
                p0=[a0, mu, sigma, tau0],
                bounds=([0, mu - 5 * sigma, 1e-3, 1e-3], [np.inf, mu + 5 * sigma, np.inf, np.inf]),
                maxfev=5000,
            )
            tail_popt, tail_ok = popt, True
        except Exception:
            pass

    n_core = int(((dt_ns >= mu - CORE_TRIM_SIGMA * sigma) & (dt_ns <= mu + CORE_TRIM_SIGMA * sigma)).sum())
    tail_frac = 1.0 - n_core / len(dt_ns) if len(dt_ns) else None

    return {
        "counts": counts, "edges": edges, "centers": centers,
        "mu": mu, "sigma_core": sigma, "core_popt": core_popt, "core_fit_ok": core_ok,
        "tail_popt": tail_popt, "tail_fit_ok": tail_ok, "tail_frac_approx": tail_frac,
    }


def find_peak_x(results_csv, pad):
    rows = [r for r in csv.DictReader(open(results_csv)) if r["pad"] == pad]
    max_n = max(int(r["n_selected"]) for r in rows)
    trustworthy = [r for r in rows if int(r["n_selected"]) >= 0.5 * max_n]
    best = max(trustworthy, key=lambda r: float(r["mean_pulse_height_mV"]))
    return int(best["x_mm"]), int(best["channel"])


def analyze_pad(cache_dir, pad, x, ch, ref_ch, timing_method, cfd_frac, min_depth_mv):
    """
    x=None -> fixed-position/dedicated-run dataset (no X<pos>Y0 tag in the
    filenames, e.g. a 2-channel timing-only capture at one laser position):
    glob every file for that channel in cache_dir directly, since the whole
    directory is already scoped to this one run/position.
    """
    if x is None:
        pad_files = sorted(glob.glob(os.path.join(cache_dir, f"*_ch{ch}_*.npz")))
        ref_files = sorted(glob.glob(os.path.join(cache_dir, f"*_ch{ref_ch}_*.npz")))
    else:
        pad_files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))
        ref_files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ref_ch}_*.npz")))
    n = min(len(pad_files), len(ref_files))

    time_fn = (lambda t, v, lo, hi, thr: cfd_time(t, v, lo, hi, cfd_frac, thr)) if timing_method == "cfd" \
        else (lambda t, v, lo, hi, thr: centroid_time(t, v, lo, hi, thr))

    dts = []
    n_pad_fired = 0
    for i in range(n):
        with np.load(pad_files[i]) as z:
            t_p, v_p = z["t"], z["v"]
        b_p = np.median(v_p[(t_p >= PAD_BASELINE_T_RANGE[0]) & (t_p <= PAD_BASELINE_T_RANGE[1])])
        t_pad, min_pad = time_fn(t_p, v_p - b_p, *PAD_SIGNAL_T_RANGE, min_depth_mv)
        if t_pad is None:
            continue
        n_pad_fired += 1

        with np.load(ref_files[i]) as z:
            t_r, v_r = z["t"], z["v"]
        b_r = np.median(v_r[(t_r >= REF_BASELINE_T_RANGE[0]) & (t_r <= REF_BASELINE_T_RANGE[1])])
        t_ref, _ = time_fn(t_r, v_r - b_r, *REF_SEARCH_T_RANGE, None)
        if t_ref is None:
            continue

        dts.append(t_pad - t_ref)

    return np.array(dts), n, n_pad_fired


def main():
    global REF_SEARCH_T_RANGE
    ap = argparse.ArgumentParser(description="HRPPD transit-time-spread (TTS) analysis vs. laser-sync reference "
                                              "(methodology per Shin et al. arXiv:2212.03208 Sec 3.3/5.3).")
    ap.add_argument("--run", required=True)
    ap.add_argument("--results-csv", default=None,
                     help="output of analyze_hrppd.py, used to find each pad's peak X and channel. "
                          "Omit for a dedicated fixed-position run and use --pad/--pad-channel instead.")
    ap.add_argument("--pad", default=None, help="pad label for a fixed-position run (with --pad-channel)")
    ap.add_argument("--pad-channel", type=int, default=None,
                     help="scope channel for a fixed-position run (no X<pos>Y0 tag in filenames); "
                          "set together with --pad to bypass --results-csv entirely")
    ap.add_argument("--x-label", default=None, help="display-only position label for a fixed-position run, e.g. '31'")
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--ref-channel", type=int, default=1)
    ap.add_argument("--timing-method", choices=["centroid", "cfd"], default="centroid",
                     help="per-event timing estimator: 'centroid' (default, Shin et al. Sec 3.3, "
                          "charge-weighted mean of the pulse) or 'cfd' (50%%-of-peak crossing, prior version)")
    ap.add_argument("--cfd-frac", type=float, default=CFD_FRAC, help="only used with --timing-method cfd")
    ap.add_argument("--min-depth-mv", type=float, default=TIMING_MIN_DEPTH_MV,
                     help="minimum baseline-subtracted pulse depth (mV) required for an event "
                          "to count toward timing; default -25.0")
    ap.add_argument("--ref-window", type=float, nargs=2, default=list(REF_SEARCH_T_RANGE))
    ap.add_argument("--laser-sigma-ps", type=float, default=None,
                     help="Shin et al. Eq 3.6: independently-measured sigma of the laser pulse width (ps). "
                          "If given, an additional laser-width-corrected sigma_core is reported: "
                          "sqrt(sigma_core^2 - sigma_LW^2). Omit if unknown -- reported sigma_core is then "
                          "the raw measured value (same caveat Shin et al. give for their own numbers).")
    ap.add_argument("--out-prefix", default=None, help="default: hrppd_tts_<run>")
    args = ap.parse_args()
    out_prefix = args.out_prefix or f"hrppd_tts_{args.run}"

    REF_SEARCH_T_RANGE = tuple(args.ref_window)

    cache_dir = os.path.join(args.cache_root, args.run)
    fixed_position = args.pad is not None and args.pad_channel is not None
    if fixed_position:
        pads = [args.pad]
        fixed_x = {args.pad: (None, args.pad_channel)}
    else:
        if not args.results_csv:
            raise SystemExit("need --results-csv, or both --pad and --pad-channel for a fixed-position run")
        pads = []
        seen = set()
        for row in csv.DictReader(open(args.results_csv)):
            if row["pad"] not in seen:
                seen.add(row["pad"])
                pads.append(row["pad"])
        fixed_x = None

    print(f"[tts] run: {args.run}  pads: {pads}  ref channel: ch{args.ref_channel}  "
          f"timing method: {args.timing_method}  min depth: {args.min_depth_mv} mV"
          f"{'  (fixed-position mode)' if fixed_position else ''}")
    if args.laser_sigma_ps is None:
        print("[tts] NOTE: no --laser-sigma-ps given -- reported sigma_core is the RAW measured value, "
              "NOT corrected for the finite laser pulse width (Shin et al. Eq 3.6 not applied).")

    summary_rows = []
    fig, axes = plt.subplots(1, len(pads), figsize=(5.2 * len(pads), 4.6), dpi=150)
    if len(pads) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for i, pad in enumerate(pads):
        x, ch = fixed_x[pad] if fixed_position else find_peak_x(args.results_csv, pad)
        x_disp = args.x_label if (fixed_position and args.x_label) else x
        dts, n_total, n_pad_fired = analyze_pad(cache_dir, pad, x, ch, args.ref_channel,
                                                 args.timing_method, args.cfd_frac, args.min_depth_mv)
        n_matched = len(dts)

        color = SLOT_COLORS[i % len(SLOT_COLORS)]
        ax = axes[i]
        ax.set_facecolor(SURFACE)

        row = {"pad": pad, "x_mm": x_disp, "channel": ch, "n_total": n_total,
               "n_pad_fired": n_pad_fired, "n_matched": n_matched, "timing_method": args.timing_method}

        if n_matched >= 30:
            dt_ns = dts * 1.0
            fit = fit_core_and_tail(dt_ns)
            mu, sigma_core = fit["mu"], fit["sigma_core"]
            sigma_core_ps = sigma_core * 1000

            sigma_core_corr_ps = None
            if args.laser_sigma_ps is not None:
                diff2 = sigma_core_ps ** 2 - args.laser_sigma_ps ** 2
                sigma_core_corr_ps = float(np.sqrt(diff2)) if diff2 > 0 else 0.0

            row.update({
                "mean_dt_ns": float(np.mean(dt_ns)), "mu_core_ns": mu,
                "sigma_core_ns": float(sigma_core), "sigma_core_ps": float(sigma_core_ps),
                "fwhm_core_ps": float(2.3548 * sigma_core_ps),
                "sigma_core_corrected_ps": sigma_core_corr_ps,
                "core_fit_ok": fit["core_fit_ok"], "tail_fit_ok": fit["tail_fit_ok"],
                "tail_frac_approx": fit["tail_frac_approx"],
                "tail_tau_ps": float(fit["tail_popt"][3] * 1000) if fit["tail_fit_ok"] else None,
            })

            centers, counts, edges = fit["centers"], fit["counts"], fit["edges"]
            ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=color, alpha=0.6, edgecolor="none")
            xs = np.linspace(edges[0], edges[-1], 400)
            if fit["core_fit_ok"]:
                ax.plot(xs, gauss(xs, *fit["core_popt"]), color=INK, linewidth=1.8, zorder=3, label="core (Gaussian)")
            if fit["tail_fit_ok"]:
                ax.plot(xs, exgauss(xs, *fit["tail_popt"]), color="#c0392b", linewidth=1.4, linestyle="--",
                         zorder=2, label="tail (ex-Gaussian)")
            legend = ax.legend(frameon=False, fontsize=7.5, loc="upper left")
            for txt in legend.get_texts():
                txt.set_color(INK)

            title_sigma = f"σ_core,corr={sigma_core_corr_ps:.0f} ps" if sigma_core_corr_ps is not None \
                else f"σ_core={sigma_core_ps:.0f} ps (uncorrected)"
            ax.set_title(f"{pad} (ch{ch} vs ch{args.ref_channel})\n"
                         f"X={x_disp} mm, n={n_matched}, {title_sigma}",
                         color=INK, fontsize=10.5, pad=10)
        else:
            row.update({"mean_dt_ns": None, "mu_core_ns": None, "sigma_core_ns": None, "sigma_core_ps": None,
                        "fwhm_core_ps": None, "sigma_core_corrected_ps": None, "core_fit_ok": False,
                        "tail_fit_ok": False, "tail_frac_approx": None, "tail_tau_ps": None})
            ax.text(0.5, 0.5, f"only {n_matched} matched events\n(insufficient statistics)",
                    ha="center", va="center", transform=ax.transAxes, color=SECONDARY)
            ax.set_title(f"{pad} (ch{ch} vs ch{args.ref_channel})", color=INK, fontsize=11)

        ax.set_xlabel(f"pad − ref {args.timing_method} time (ns)", color=INK, fontsize=10)
        ax.set_ylabel("events", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=SECONDARY, labelsize=8.5)

        summary_rows.append(row)
        if row["sigma_core_ps"] is not None:
            extra = f" corrected={row['sigma_core_corrected_ps']:.1f}ps" if row["sigma_core_corrected_ps"] is not None else ""
            print(f"[tts] {pad}: X={x_disp}mm n_matched={n_matched}/{n_total} "
                  f"sigma_core={row['sigma_core_ps']:.1f}ps{extra} "
                  f"tail_frac~{row['tail_frac_approx']:.3f} tail_fit_ok={row['tail_fit_ok']}")
        else:
            print(f"[tts] {pad}: X={x_disp}mm n_matched={n_matched}/{n_total} -- too few events")

    fig.tight_layout()
    out_png = f"{out_prefix}.png"
    fig.savefig(out_png, facecolor=SURFACE)
    print(f"[tts] wrote {out_png}")

    out_csv = f"{out_prefix}.csv"
    fieldnames = ["pad", "x_mm", "channel", "timing_method", "n_total", "n_pad_fired", "n_matched",
                  "mean_dt_ns", "mu_core_ns", "sigma_core_ns", "sigma_core_ps", "fwhm_core_ps",
                  "sigma_core_corrected_ps", "core_fit_ok", "tail_fit_ok", "tail_frac_approx", "tail_tau_ps"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[tts] wrote {out_csv}")


if __name__ == "__main__":
    main()
