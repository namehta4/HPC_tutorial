#!/usr/bin/env python3
"""Overlay all four stages' tracked kernels on a single roofline chart.

Usage:
    python3 plot_roofline.py --peak-bw 1500 --peak-flops-fp64 9.7 \
        --peak-flops-fp32 156 --peak-flops-fp16 312 \
        --csv stage1_naive/profiles/stage1_vqe_roofline.csv \
        --csv stage2_gpu_native/profiles/stage2_vqe_fused_roofline.csv \
        --csv stage3_memory_opt/profiles/stage3_vqe_fp64_roofline.csv \
        --csv stage4_compute_opt/profiles/stage4_vqe_roofline.csv \
        --out roofline_overlay.png

Peak bandwidth/FLOPs values should come from stage0_calibration/RESULTS.md
(the MEASURED numbers for your GPU), not vendor spec sheets -- that's the
entire point of stage0.

CSV files are produced by:
    ncu --import <report>.ncu-rep --csv --page raw > <report>.csv

This script looks for these ncu metric columns (present in --set roofline
or --set full reports) to compute each kernel's (arithmetic intensity,
achieved GFLOP/s) point:
    - "Kernel Name" (or "Kernel Name " with trailing space, some ncu versions)
    - "dram__bytes.sum" (total DRAM bytes moved) -- used as the memory-traffic
      denominator for arithmetic intensity
    - "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum" and similar FMA
      counters -- used as a FLOP-count numerator. ncu's exact column naming
      varies by version and by which --set was used; if this script can't
      find a usable FLOP column for your ncu version, pass --manual-points
      (see --help) to supply (label, AI, GFLOPs) triples directly instead of
      parsing a CSV -- this keeps the plot usable even if ncu's schema
      differs from what we tested against.
"""
from __future__ import annotations

import argparse
import csv as csv_module
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ncu's exact metric column names have shifted across versions, and (as of
# the ncu version this was tested against, 2026.1.0.0) `--set roofline`
# does NOT export a direct "total bytes moved" column -- only a rate
# (dram__bytes.sum.per_second, in GB/s) and a duration
# (gpu__time_duration.sum, in microseconds). Total bytes = rate * duration.
# We try a short list of known aliases for each quantity and use whichever
# is present, so this script degrades gracefully rather than crashing
# outright if your ncu version's schema differs slightly.
KERNEL_NAME_COLUMNS = ["Kernel Name", "Kernel Name "]
DRAM_RATE_COLUMNS = ["dram__bytes.sum.per_second"]  # GB/s
DRAM_BYTES_COLUMNS = ["dram__bytes.sum"]  # some ncu versions/sets DO export this directly; prefer it if present
DURATION_COLUMNS = ["gpu__time_duration.sum", "Duration"]  # microseconds
FLOP_COLUMNS_FP64 = [
    "derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2",  # already x2 for FMA=2 FLOPs
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
]
FLOP_COLUMNS_FP32 = [
    "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
]


def _find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def parse_ncu_csv(path: str) -> list[dict]:
    """Best-effort extraction of (kernel_name, bytes_moved, flops, duration_s)
    per kernel invocation from an `ncu --csv --page raw` export. Returns a
    list of dicts; entries where required columns are missing are skipped
    with a warning rather than crashing the whole plot.

    NOTE on units: `ncu --csv --page raw` prepends a units row (e.g. "us",
    "Gbyte/s", "inst") right after the header, before any real data rows.
    We detect and skip it by checking whether the duration/name field looks
    numeric/non-empty for a real kernel invocation.
    """
    results = []
    with open(path, newline="") as f:
        reader = csv_module.DictReader(f)
        fieldnames = reader.fieldnames or []

        name_col = _find_column(fieldnames, KERNEL_NAME_COLUMNS)
        bytes_col = _find_column(fieldnames, DRAM_BYTES_COLUMNS)
        rate_col = _find_column(fieldnames, DRAM_RATE_COLUMNS)
        dur_col = _find_column(fieldnames, DURATION_COLUMNS)
        flop64_col = _find_column(fieldnames, FLOP_COLUMNS_FP64)
        flop32_col = _find_column(fieldnames, FLOP_COLUMNS_FP32)

        if name_col is None:
            print(f"WARNING: {path}: no kernel-name column found among {KERNEL_NAME_COLUMNS}; skipping file")
            return results
        if dur_col is None or (bytes_col is None and rate_col is None):
            print(f"WARNING: {path}: missing bytes/rate/duration columns; this ncu version's schema differs. "
                  "Use --manual-points instead for this file.")
            return results

        for row in reader:
            name = row.get(name_col, "")
            if not name:  # skip the units row and any blank-name row
                continue
            try:
                duration_us = float(row[dur_col].replace(",", "")) if row[dur_col] else 0.0
                if bytes_col and row.get(bytes_col):
                    bytes_moved = float(row[bytes_col].replace(",", ""))
                elif rate_col and row.get(rate_col):
                    # rate is GB/s, duration is microseconds -> bytes = GB/s * 1e9 * (us * 1e-6)
                    rate_gbs = float(row[rate_col].replace(",", ""))
                    bytes_moved = rate_gbs * 1e9 * (duration_us * 1e-6)
                else:
                    bytes_moved = 0.0

                flops = 0.0
                if flop64_col and row.get(flop64_col):
                    flops += float(row[flop64_col].replace(",", ""))
                if flop32_col and row.get(flop32_col):
                    flops += float(row[flop32_col].replace(",", ""))
            except (ValueError, KeyError):
                continue

            if bytes_moved <= 0 or duration_us <= 0:
                continue

            results.append({
                "name": name,
                "bytes": bytes_moved,
                "flops": flops,
                "duration_s": duration_us * 1e-6,
            })
    return results


def compute_ai_and_gflops(entry: dict) -> tuple[float, float]:
    """Arithmetic intensity (FLOP/byte) and achieved GFLOP/s for one kernel invocation."""
    ai = entry["flops"] / entry["bytes"] if entry["bytes"] > 0 else 0.0
    gflops = (entry["flops"] / entry["duration_s"]) / 1e9 if entry["duration_s"] > 0 else 0.0
    return ai, gflops


def plot_roofline(
    stage_points: dict[str, list[tuple[str, float, float]]],
    peak_bw_gbs: float,
    peak_flops: dict[str, float],
    out_path: str,
):
    """stage_points: {stage_label: [(kernel_label, AI, achieved_GFLOPs), ...]}
    peak_flops: {"fp64": ..., "fp32": ..., "fp16": ...} in GFLOP/s."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ai_range = np.logspace(-3, 3, 200)

    for precision, peak in sorted(peak_flops.items(), key=lambda kv: kv[1]):
        ridge_ai = peak / peak_bw_gbs
        roofline = np.minimum(peak, ai_range * peak_bw_gbs)
        ax.plot(ai_range, roofline, "--", alpha=0.5, label=f"{precision.upper()} roofline (peak={peak:.1f} GFLOP/s)")
        ax.axvline(ridge_ai, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)

    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(stage_points), 1)))
    markers = ["o", "s", "^", "D", "v", "P"]

    for (stage_label, points), color in zip(stage_points.items(), colors):
        for i, (kernel_label, ai, gflops) in enumerate(points):
            marker = markers[i % len(markers)]
            ax.scatter(
                ai, gflops, color=color, marker=marker, s=120, edgecolors="black", linewidths=0.8, zorder=5,
                label=f"{stage_label}: {kernel_label}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (GFLOP/s)", fontsize=12)
    ax.set_title("Roofline: Stage 1-4 Kernel Progression\n(gate_apply, expval_reduce, policy_forward)", fontsize=13)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", default=[], help="ncu --csv --page raw export; repeatable, one per stage.")
    ap.add_argument("--stage-label", action="append", default=[],
                     help="Label for each --csv, in the same order (default: derived from filename).")
    ap.add_argument("--manual-points", type=str, default=None,
                     help="Path to a JSON file of {stage_label: [[kernel_label, AI, GFLOPs], ...]} "
                     "to use INSTEAD of parsing ncu CSVs (for ncu schema versions this script's "
                     "column-name heuristics don't match).")
    ap.add_argument("--peak-bw", type=float, required=True, help="Measured peak memory bandwidth, GB/s (from stage0).")
    ap.add_argument("--peak-flops-fp64", type=float, default=None, help="Measured peak FP64 GFLOP/s (from stage0).")
    ap.add_argument("--peak-flops-fp32", type=float, default=None, help="Measured peak FP32/TF32 GFLOP/s (from stage0).")
    ap.add_argument("--peak-flops-fp16", type=float, default=None, help="Measured peak FP16 GFLOP/s (from stage0).")
    ap.add_argument("--out", type=str, default="roofline_overlay.png")
    ap.add_argument("--max-kernels-per-stage", type=int, default=5,
                     help="Cap how many distinct kernel invocations to plot per stage's CSV, to avoid clutter.")
    args = ap.parse_args()

    peak_flops = {}
    if args.peak_flops_fp64:
        peak_flops["fp64"] = args.peak_flops_fp64 * 1000  # TFLOP/s -> GFLOP/s if given in TFLOP/s; see note below
    if args.peak_flops_fp32:
        peak_flops["fp32"] = args.peak_flops_fp32 * 1000
    if args.peak_flops_fp16:
        peak_flops["fp16"] = args.peak_flops_fp16 * 1000
    if not peak_flops:
        print("ERROR: at least one of --peak-flops-fp64/--peak-flops-fp32/--peak-flops-fp16 is required "
              "(values from stage0_calibration/RESULTS.md, in TFLOP/s).")
        sys.exit(1)

    stage_points: dict[str, list[tuple[str, float, float]]] = {}

    if args.manual_points:
        import json

        with open(args.manual_points) as f:
            raw = json.load(f)
        for stage_label, points in raw.items():
            stage_points[stage_label] = [(p[0], p[1], p[2]) for p in points]
    else:
        if not args.csv:
            print("ERROR: provide --csv (one or more) or --manual-points.")
            sys.exit(1)
        for i, csv_path in enumerate(args.csv):
            label = args.stage_label[i] if i < len(args.stage_label) else csv_path.split("/")[0]
            entries = parse_ncu_csv(csv_path)
            points = []
            seen_names = set()
            for entry in entries:
                if entry["name"] in seen_names:
                    continue
                if len(points) >= args.max_kernels_per_stage:
                    break
                seen_names.add(entry["name"])
                ai, gflops = compute_ai_and_gflops(entry)
                if ai > 0 and gflops > 0:
                    points.append((entry["name"], ai, gflops))
            if points:
                stage_points[label] = points
            else:
                print(f"WARNING: no usable data points extracted from {csv_path}")

    if not stage_points:
        print("ERROR: no data points to plot. Check your CSV files or use --manual-points.")
        sys.exit(1)

    plot_roofline(stage_points, args.peak_bw, peak_flops, args.out)


if __name__ == "__main__":
    main()
