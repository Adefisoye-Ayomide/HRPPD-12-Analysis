"""
analyze_hrppd.py
=================
HRPPD #12 X-scan analysis.

Reads the NPZ cache built by build_cache_hrppd.py and computes, per X
position (1 mm steps) and per channel, the mean pulse height (peak negative
amplitude) of the waveform.

Baseline subtraction & signal windowing:
    Per event, per channel: baseline = median(v) over t in [-10, 0] ns
    (before the expected pulse arrives; median chosen over mean for
    robustness to the rare (~1% of events, measured directly -- see
    hrppd_diag_baseline_median_vs_mean_*.png) stray dark-count pulse landing
    in the baseline window). Subtracted from the whole trace, then the pulse
    height is the minimum of the baseline-subtracted trace restricted to t
    in [40, 65] ns -- the true-pulse arrival window, measured directly from
    RMS(t) across ~1500 events per channel on run 260803XY: two channels
    rose sharply at ~48ns and were back to baseline noise by ~56-58ns; the
    third peaked later (~52-53ns) with a tail persisting to ~63-65ns.
    [40,65]ns covered all three with margin. This excludes afterpulsing
    (ion-feedback afterpulses observed as late as ~120ns) and unrelated
    early excursions from contaminating the pulse-height measurement.
    Re-validate these windows (--baseline-window / --signal-window) if the
    electronics/timing setup changes.

Noise floor (--control-window, default [10, 35] ns):
    A min-search over a signal-free window is NOT zero -- it's a positive
    extreme-value statistic that converges to roughly sigma*sqrt(2 ln N) for
    N samples, not to 0. [10, 35] ns is the same width as the signal window,
    sits after the baseline window and well before the true pulse onset
    (~44-48 ns), and was checked empirically (RMS(t) binned in 20-25ns
    chunks from -30 to 166 ns across 3 runs/pads) to be as flat/quiet as
    anywhere in the trace -- flatter than the post-pulse region, which still
    carries pulse-tail/afterpulse contamination out to ~90 ns. ctrl_mean_mV/
    ctrl_median_mV report this floor per (position, pad), UNCONDITIONED on
    gating, so you can see directly whether --threshold sits comfortably
    above it (it should -- a threshold below this floor selects close to
    100% of pure noise, not real hits).

Clipping / rail detection (--clip-eps, --rail-min-repeats):
    A channel run close to its scope's full-scale range clips on its
    biggest real pulses. Physics doesn't produce the identical peak
    amplitude, to a fraction of an mV, across many independent laser shots
    -- so if the same windowed-min value (binned to --clip-eps mV) recurs
    at least --rail-min-repeats times among a pad's own-threshold-crossing
    events at one position, that value is flagged as a rail and those
    events are EXCLUDED from that pad's mean/sem/pass_rate (counted in
    n_clipped, never corrected or extrapolated). This only ever excludes a
    pad's own contaminated events from its own statistics -- it doesn't
    touch other pads' numbers for the same events.

Channel -> pad mapping and event selection are run-specific (cables get
re-wired between runs) -- set via --channel-map and --gate-pads. ch1 (laser
sync/trigger reference) is never part of the pad set.

Event selection logic:
    All four scope channels are captured simultaneously per laser trigger
    (same timestamp to within a few ms of file-write jitter), so the n-th
    file (sorted by timestamp) for each channel at a given X is the same
    physical event.

    An event is kept for the pulse-height average if AT LEAST ONE of the
    --gate-pads crosses the threshold (i.e. is a real laser hit, not just
    baseline noise). Once kept, that event's actual pulse height is recorded
    for ALL pads in --channel-map -- including ones that did not individually
    cross threshold. This avoids censoring/truncation bias: a channel whose
    response is falling as the laser moves away from it should show its true
    (small) pulse height, not be silently dropped because it didn't
    self-trigger. Events where none of the gate pads fire (pure dark/noise)
    are excluded from the mean.

    A pad should be left out of --gate-pads if it shows a channel-specific
    artifact that isn't a real, position-dependent response (e.g. run
    260803XY's P12/ch3 crossed threshold on ~100% of events at every X
    position, including 70+ mm from its own response region -- consistent
    with electrical crosstalk timed to the trigger rather than a real
    detector signal; including it in the gate made the OR-selection pass on
    ~100% of all events everywhere, i.e. no gating at all).

    Threshold default is -25 mV, not -5 mV: measured noise floors (see
    ctrl_mean_mV column) across every pad/run analyzed so far run ~6-16 mV,
    so a -5mV cut sat below or barely above the noise floor on several
    channels -- selecting a large fraction of pure noise as "real hits" and
    (via the extreme-value effect above) biasing means upward in a way that
    changes with position. -25mV sits comfortably above every measured
    floor to date.

Usage:
    python analyze_hrppd.py --run RUN --channel-map "2:P13,3:P9,4:P5" \\
        --gate-pads "P13,P9,P5" [--cache-root ROOT] [--threshold MV]
"""

import os, glob, re, argparse
from collections import Counter
from multiprocessing import Pool, cpu_count
import numpy as np

CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"
RUN        = "260804XY"
THRESHOLD_MV = -25.0  # HRPPD signal is negative; "passes" means v.min() <= this

BASELINE_T_RANGE = (-10.0, 0.0)  # ns, pre-pulse quiet window (median, robust to rare dark counts)
SIGNAL_T_RANGE   = (40.0, 65.0)  # ns, true-pulse arrival window (see module docstring)
CONTROL_T_RANGE  = (10.0, 35.0)  # ns, signal-free noise-floor reference, same width as SIGNAL_T_RANGE

CLIP_EPS_MV      = 0.5   # mV bin width for detecting a recurring (rail) extremum
RAIL_MIN_REPEATS = 5     # min recurrences at one position to flag a value as a rail

CHANNEL_MAP = "2:P13,3:P9,4:P5"  # ch -> pad, in low-X -> high-X spatial order
GATE_PADS   = "P13,P9,P5"        # pads OR'd together to select real-hit events

CHANNEL_PAD = {}   # populated from --channel-map at startup
PAD_ORDER   = []   # populated from --channel-map at startup (preserves given order)
GATE_PAD_LIST = []  # populated from --gate-pads at startup

FNAME_RE = re.compile(r"X(\d+)Y0_ch(\d+)_(\d+)\.npz$")


def parse_channel_map(s):
    """'2:P13,3:P9,4:P5' -> {2: 'P13', 3: 'P9', 4: 'P5'}, order preserved."""
    pairs = [p.strip() for p in s.split(",") if p.strip()]
    d = {}
    for p in pairs:
        ch_str, pad = p.split(":")
        d[int(ch_str.strip())] = pad.strip()
    return d


def find_positions(cache_dir, any_channel):
    xs = set()
    for f in glob.glob(os.path.join(cache_dir, f"*Y0_ch{any_channel}_*.npz")):
        m = FNAME_RE.search(f)
        if m:
            xs.add(int(m.group(1)))
    return sorted(xs)


def load_channel_mins(cache_dir, x, ch):
    """
    Return (mins, ctrl_mins): per-event, baseline-subtracted min(v) for one
    (position, channel), sorted by timestamp -- in SIGNAL_T_RANGE (the real
    measurement) and in CONTROL_T_RANGE (the signal-free noise-floor
    reference, same estimator, same width).
    """
    files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))
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
    """
    vals_mv: raw (negative-going) windowed-min values for events that
    individually cross threshold on this pad. If the same value (binned to
    eps mV) recurs >= min_repeats times, flag it as a rail/clip value and
    return (rail_value_mV, boolean mask of events AT that rail). Otherwise
    (None, all-False mask).
    """
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
    cache_dir, x, threshold = args
    per_ch, per_ch_ctrl = {}, {}
    for ch, pad in CHANNEL_PAD.items():
        per_ch[pad], per_ch_ctrl[pad] = load_channel_mins(cache_dir, x, ch)

    n = min(len(per_ch[pad]) for pad in PAD_ORDER)
    if n == 0:
        return {"x": x, "n_total": 0, "n_selected": 0,
                "mean": {p: np.nan for p in PAD_ORDER},
                "sem":  {p: np.nan for p in PAD_ORDER},
                "pass_rate": {p: np.nan for p in PAD_ORDER},
                "ctrl_mean": {p: np.nan for p in PAD_ORDER},
                "ctrl_median": {p: np.nan for p in PAD_ORDER},
                "n_clipped": {p: 0 for p in PAD_ORDER},
                "clip_value": {p: np.nan for p in PAD_ORDER}}

    counts = {p: len(per_ch[p]) for p in PAD_ORDER}
    if len(set(counts.values())) != 1:
        # Simultaneous multi-ch capture should give equal counts per channel;
        # if not, align on the first n (sorted by timestamp) and note it.
        pass

    mat = np.stack([per_ch[p][:n] for p in PAD_ORDER], axis=1)  # (n_events, n_pads)
    ctrl_mat = np.stack([per_ch_ctrl[p][:n] for p in PAD_ORDER], axis=1)
    passed = mat <= threshold                                    # per-channel pass
    gate_idx = [PAD_ORDER.index(p) for p in GATE_PAD_LIST]
    selected = passed[:, gate_idx].any(axis=1)                   # OR across gate pads only

    result = {"x": x, "n_total": n, "n_selected": int(selected.sum()),
              "counts": counts, "mean": {}, "sem": {}, "pass_rate": {},
              "ctrl_mean": {}, "ctrl_median": {}, "n_clipped": {}, "clip_value": {}}

    sel_mat = mat[selected]
    for j, pad in enumerate(PAD_ORDER):
        # Clip/rail detection: among THIS pad's own threshold-crossing events
        # at this position (independent of the gate), flag a recurring
        # extremum and drop those events from this pad's own stats only.
        own_pass_idx = np.where(mat[:, j] <= threshold)[0]
        rail_val, rail_mask_own = detect_rail(mat[own_pass_idx, j]) if own_pass_idx.size else (None, np.array([], bool))
        clipped_global = np.zeros(n, dtype=bool)
        if rail_val is not None:
            clipped_global[own_pass_idx[rail_mask_own]] = True
        result["n_clipped"][pad] = int(clipped_global.sum())
        result["clip_value"][pad] = rail_val if rail_val is not None else float("nan")

        sel_not_clipped = selected & (~clipped_global)
        vals = -mat[sel_not_clipped, j]  # positive-magnitude pulse height (mV)
        if vals.size > 0:
            result["mean"][pad] = float(np.mean(vals))
            result["sem"][pad]  = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            result["pass_rate"][pad] = float(passed[sel_not_clipped, j].mean())
        else:
            result["mean"][pad] = np.nan
            result["sem"][pad] = np.nan
            result["pass_rate"][pad] = np.nan

        # Noise floor: unconditioned on gating, so it reflects this channel's
        # general behavior, not just what happened to be selected.
        ctrl_vals = -ctrl_mat[:, j]
        result["ctrl_mean"][pad] = float(np.mean(ctrl_vals))
        result["ctrl_median"][pad] = float(np.median(ctrl_vals))

    return result


def polarity_check(cache_dir, x, ch, pad, s_lo, s_hi):
    """
    Startup diagnostic only -- does not change how ph is computed (always
    -(min(v-b)), i.e. negative-going, per the module's convention). Reports
    both lobes on one representative event so a wiring-swapped channel
    (a cabling property, not a detector property) shows up as a printed
    warning instead of silently corrupting that channel's numbers.
    """
    files = sorted(glob.glob(os.path.join(cache_dir, f"*X{x}Y0_ch{ch}_*.npz")))
    if not files:
        return None
    with np.load(files[0]) as z:
        t, v = z["t"], z["v"]
    b_lo, b_hi = BASELINE_T_RANGE
    baseline = np.median(v[(t >= b_lo) & (t <= b_hi)])
    m = (t >= s_lo) & (t <= s_hi)
    w = v[m] - baseline
    neg, pos = float(-w.min()), float(w.max())
    verdict = "neg" if neg > pos else "pos"
    flag = "" if verdict == "neg" else "   <<< looks POSITIVE-going -- check wiring/config for this pad"
    print(f"[analyze]   {pad} (ch{ch}): -min={neg:7.2f} mV, +max={pos:7.2f} mV "
          f"-> looks '{verdict}'{flag}")
    return verdict


def main():
    global BASELINE_T_RANGE, SIGNAL_T_RANGE, CONTROL_T_RANGE, CHANNEL_PAD, PAD_ORDER, GATE_PAD_LIST
    ap = argparse.ArgumentParser(description="Analyze HRPPD X-scan pulse heights from NPZ cache.")
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--threshold", type=float, default=THRESHOLD_MV,
                     help="mV threshold on baseline-subtracted, windowed min(v); default -25.0")
    ap.add_argument("--baseline-window", type=float, nargs=2, default=list(BASELINE_T_RANGE),
                     metavar=("LO_NS", "HI_NS"), help="pre-pulse baseline window in ns")
    ap.add_argument("--signal-window", type=float, nargs=2, default=list(SIGNAL_T_RANGE),
                     metavar=("LO_NS", "HI_NS"), help="true-pulse search window in ns")
    ap.add_argument("--control-window", type=float, nargs=2, default=list(CONTROL_T_RANGE),
                     metavar=("LO_NS", "HI_NS"), help="signal-free noise-floor reference window in ns")
    ap.add_argument("--channel-map", default=CHANNEL_MAP,
                     help="scope channel -> pad name, e.g. '2:P13,3:P9,4:P5' "
                          "(order = low-X -> high-X spatial order)")
    ap.add_argument("--gate-pads", default=GATE_PADS,
                     help="comma-separated pad names OR'd together for event selection, "
                          "e.g. 'P13,P9,P5'. Leave out a pad if it shows a non-positional "
                          "artifact (see module docstring).")
    ap.add_argument("--out-csv", default=None,
                     help="default: hrppd_xscan_results_<run>.csv")
    args = ap.parse_args()
    if args.out_csv is None:
        args.out_csv = f"hrppd_xscan_results_{args.run}.csv"

    BASELINE_T_RANGE = tuple(args.baseline_window)
    SIGNAL_T_RANGE = tuple(args.signal_window)
    CONTROL_T_RANGE = tuple(args.control_window)
    CHANNEL_PAD = parse_channel_map(args.channel_map)
    PAD_ORDER = list(CHANNEL_PAD.values())
    GATE_PAD_LIST = [p.strip() for p in args.gate_pads.split(",") if p.strip()]
    for p in GATE_PAD_LIST:
        if p not in PAD_ORDER:
            raise SystemExit(f"--gate-pads pad '{p}' not found in --channel-map pads {PAD_ORDER}")

    print(f"[analyze] run: {args.run}")
    print(f"[analyze] channel map: {CHANNEL_PAD}  (pad order: {PAD_ORDER})")
    print(f"[analyze] gate pads (OR'd for event selection): {GATE_PAD_LIST}")
    print(f"[analyze] threshold: {args.threshold} mV")
    print(f"[analyze] baseline window: {BASELINE_T_RANGE} ns, signal window: {SIGNAL_T_RANGE} ns, "
          f"control window: {CONTROL_T_RANGE} ns")

    cache_dir = os.path.join(args.cache_root, args.run)
    xs = find_positions(cache_dir, next(iter(CHANNEL_PAD)))
    print(f"[analyze] {len(xs)} X positions found in {cache_dir}")

    print("[analyze] polarity check (one waveform per pad, at the first X position; "
          "ph is always computed as -(min(v-b)) regardless of this result):")
    for ch, pad in CHANNEL_PAD.items():
        polarity_check(cache_dir, xs[0], ch, pad, *SIGNAL_T_RANGE)

    tasks = [(cache_dir, x, args.threshold) for x in xs]
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(process_position, tasks)

    results.sort(key=lambda r: r["x"])

    pad_channel = {pad: ch for ch, pad in CHANNEL_PAD.items()}
    pad_gated = {pad: (pad in GATE_PAD_LIST) for pad in PAD_ORDER}

    for pad in PAD_ORDER:
        total_clipped = sum(r["n_clipped"][pad] for r in results)
        if total_clipped:
            rail_positions = [(r["x"], r["n_clipped"][pad], r["clip_value"][pad])
                               for r in results if r["n_clipped"][pad] > 0]
            worst = max(rail_positions, key=lambda t: t[1])
            print(f"[analyze]   [CLIP] {pad}: {total_clipped} events excluded as clipped/railed "
                  f"across {len(rail_positions)} positions (worst: x={worst[0]}mm, "
                  f"{worst[1]} events pinned at {worst[2]:.1f} mV)")

    import csv
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_mm", "pad", "channel", "gated", "mean_pulse_height_mV", "sem_mV",
                    "pass_rate", "n_selected", "n_total", "ctrl_mean_mV", "ctrl_median_mV",
                    "n_clipped", "clip_value_mV"])
        for r in results:
            for pad in PAD_ORDER:
                w.writerow([r["x"], pad, pad_channel[pad], pad_gated[pad],
                            r["mean"][pad], r["sem"][pad],
                            r["pass_rate"][pad], r["n_selected"], r["n_total"],
                            r["ctrl_mean"][pad], r["ctrl_median"][pad],
                            r["n_clipped"][pad], r["clip_value"][pad]])
    print(f"[analyze] wrote {args.out_csv}")

    npy_path = f"hrppd_xscan_results_{args.run}.npy"
    np.save(npy_path, np.array(results, dtype=object), allow_pickle=True)
    print(f"[analyze] wrote {npy_path} (raw results, for plotting)")


if __name__ == "__main__":
    main()
