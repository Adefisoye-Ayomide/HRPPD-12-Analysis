# HRPPD #12 Analysis

Analysis and plotting code for characterizing HRPPD unit #12, part of ongoing picosecond photodetector R&D supporting the LHCb Upgrade II photon-detection program.

This repository contains **code only** — waveform reading, analysis, and plotting scripts. Raw data, generated figures, and result files are kept out of version control (see below).

## What's in this repo

```
HRPPD_#12_Analysis/
├── tekwfm.py                                  # Tektronix waveform file (.wfm) reader
├── pipeline_scripts/                          # Production analysis pipeline (8 scripts)
├── plot_hrppd_xscan.py                        # Plot X-position scan results
├── plot_xscan_comparison_260826_827_829.py    # Compare X-scan results across run dates
├── plot_waveform_overlays.py                  # Overlay multiple waveforms for comparison
├── plot_raw_waveforms.py                      # Plot raw digitized waveforms
├── plot_individual_waveforms_260812.py        # Plot individual waveforms for a specific run
├── plot_ph_distribution.py                    # Pulse-height distribution histogram
├── plot_peak_time_distribution.py             # Peak-time distribution histogram
├── plot_260819XY_peaks_boundaries.py          # Peak/boundary detection, specific XY scan run
├── plot_260817XY_peaks_boundaries.py          # Peak/boundary detection, specific XY scan run
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

> Confirm actual imports — likely `numpy`, `scipy`, `matplotlib`, `pandas`.

## What's NOT in this repo

| Excluded | Why |
|---|---|
| `RAW_DATA/` | Raw oscilloscope waveform files — too large for git |
| `hrppd_*scan_results_*.csv` / `.npy` | Processed per-run result files |
| `hrppd_*scan_*_pulse_height.png` | Generated plots |
| `paper_figures/` | Publication figures |
| `overlay_plots/` | Generated overlay images (18 files) |
| `HRPPD_Xscan_Results_Summary.pptx` | Summary slides |
| `__pycache__/` | Compiled Python bytecode, regenerates automatically |


*** Part of the Syracuse University high-energy physics instrumentation group's photodetector R&D, supporting LHCb Upgrade II detector commissioning. ***

## Contact
Ayomide Matthew Adefisoye
Physics PhD Candidate, Syracuse University
matthewayomide3@gmail.com
