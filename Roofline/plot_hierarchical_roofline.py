#!/usr/bin/env python3
"""Hierarchical (L1 / L2 / HBM) roofline plots from Nsight Compute reports.

Every kernel gets one point per memory level, all at the same achieved
GFLOP/s but with a different arithmetic intensity (FLOPs / bytes moved at
that level). The gap between a kernel's L1, L2 and HBM points shows where
its data actually lives -- e.g. stage3's whole-circuit-in-shared-memory
kernels should show high HBM AI but low L1 AI.

The points and ceilings use exactly the metrics and formulas of Nsight
Compute's own "GPU Speed Of Light Hierarchical Roofline" sections
(sections/SpeedOfLight_Hierarchical*RooflineChart.section in the ncu
install), all of which `--set roofline` already collects -- so these PNGs
show the same numbers as the ncu GUI's roofline charts:
    L1 traffic   l1tex__lsu_writeback_active_mem_lg.sum.per_second * 128   (global/local; _lgds on CC >= 9.0)
    L2 traffic   lts__lts2xbar_cycles_active.sum.per_second * 32
    HBM traffic  dram__bytes.sum.per_second
    FLOP/s       smsp__sass_thread_inst_executed_op_{d,f}{add,mul,fma}_pred_on.sum.per_cycle_elapsed
                 (FMA x2) * smsp__cycles_elapsed.avg.per_second, plus tensor-core
                 sm__ops_path_tensor_src_*.sum (already in FLOPs)
    ceilings     <metric>.peak_sustained * <unit>__cycles_elapsed.avg.per_second, i.e. theoretical
                 peaks at the clocks ncu locked during profiling
Differences from the GUI, by design: the GUI draws one chart per precision
(double/single/half/tensor) and one point per kernel *launch*; here the
double, single and tensor charts' FLOPs are summed into one point and
launches of the same kernel are aggregated (total FLOPs / total time). For
a single-precision kernel launched once the numbers are identical.

The GUI's *half*-precision (CUDA-core HADD/HMUL/HFMA) chart is left out by
default: on Ampere, nvcc uses `HFMA2.MMA Rx, -RZ, RZ, imm` as a plain
register move, which the hfma counter can't tell apart from real math.
Nothing in this tutorial does CUDA-core fp16 arithmetic (its fp16 work runs
on tensor cores, counted above), so those counts are pure artifact -- 65% of
stage2's fused-VQE "FLOPs" before this was excluded. Pass --include-half for
code that really does fp16 math on CUDA cores.

stage0's *measured* HBM bandwidth is drawn as a dashed line next to ncu's
theoretical HBM ceiling, so the gap between the two is visible.

Every stage's profile.sh calls this automatically at the end, so the plots
are regenerated on every profiling run. To run it by hand:

    # Per-report plots (accepts .ncu-rep -- exported to CSV via `ncu --import`
    # next to the report -- or an already-exported .csv)
    python3 plot_hierarchical_roofline.py stage1_naive/profiles/stage1_vqe_roofline.ncu-rep

    # Cross-stage comparison (one aggregate L1/L2/HBM point triplet per stage)
    python3 plot_hierarchical_roofline.py --compare vqe --compare gqe

Per-report output lands in <report dir>/roofline_plots/:
    <tag>_all.png       every kernel at L1 (o), L2 (s) and HBM (v)
    <tag>_hbm.png, _l2.png, _l1.png   one level, one point per kernel
    <tag>_overview.png  whole workload aggregated into one point per level
    <tag>_kernels.csv   the per-kernel table behind the plots
--compare writes roofline_hierarchical_{vqe,gqe}.png at the repo root.

Caveat: the overview/--compare points sum kernel time only, so they don't
show launch gaps or host syncs (stage1's real bottleneck) -- that's what the
Nsight Systems timeline is for.
"""
from __future__ import annotations

import argparse
import csv as csv_module
import json
import os
import re
import statistics
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- ncu metrics (all in `--set roofline`) ----------------------------------

LEVELS = ["L1", "L2", "HBM"]
# level -> [(achieved rate column, bytes per unit, ceiling per-cycle column, clock column)];
# the first alternative present in the report is used.
TRAFFIC_METRICS = {
    "L1": [(f"l1tex__lsu_writeback_active_{m}.sum.per_second", 128,
            f"l1tex__lsu_writeback_active_{m}.sum.peak_sustained", "l1tex__cycles_elapsed.avg.per_second")
           for m in ("mem_lg", "mem_lgds")],
    "L2": [("lts__lts2xbar_cycles_active.sum.per_second", 32,
            "lts__lts2xbar_cycles_active.sum.peak_sustained", "lts__cycles_elapsed.avg.per_second")],
    "HBM": [("dram__bytes.sum.per_second", 1,
             "dram__bytes.sum.peak_sustained", "dram__cycles_elapsed.avg.per_second")],
}
# CUDA-core FLOPs: [(per-cycle column, FLOPs per thread instruction), ...] alternatives per op.
_SASS = "smsp__sass_thread_inst_executed_op_{}_pred_on.sum.per_cycle_elapsed"
CC_FLOP_METRICS = [
    [(_SASS.format("dadd"), 1)], [(_SASS.format("dmul"), 1)], [(_SASS.format("dfma"), 2)],
    [(_SASS.format("fadd"), 1)], [(_SASS.format("fmul"), 1)], [(_SASS.format("ffma"), 2)],
    [(_SASS.format("fadd2"), 2)], [(_SASS.format("fmul2"), 2)], [(_SASS.format("ffma2"), 4)],
]
# Packed half2 ops (HFMA2 = 4 FLOPs, HADD2/HMUL2 = 2); only with --include-half (see docstring).
HALF_FLOP_METRICS = [
    [(_SASS.format("hadd"), 2), ("derived__smsp__sass_thread_inst_executed_op_hadd_pred_on_x2", 1)],
    [(_SASS.format("hmul"), 2), ("derived__smsp__sass_thread_inst_executed_op_hmul_pred_on_x2", 1)],
    [(_SASS.format("hfma"), 4)],
]
SMSP_CLOCK = "smsp__cycles_elapsed.avg.per_second"
SM_CLOCK = "sm__cycles_elapsed.avg.per_second"
# Tensor-core FLOPs (sm__ops_path_tensor_* counts FMA as 2 -- checked against
# 2*M*N*K for stage4's padded GQE GEMMs). `--set roofline` exports these
# split into _sparsity_off/_on (fp64 has no split).
TC_FLOP_METRICS = ["sm__ops_path_tensor_src_fp64", "sm__ops_path_tensor_src_tf32_dst_fp32",
                   "sm__ops_path_tensor_src_fp16_dst_fp16", "sm__ops_path_tensor_src_fp16_dst_fp32",
                   "sm__ops_path_tensor_src_bf16_dst_fp32"]
TIME_METRIC = "gpu__time_duration.sum"

# name -> (per-cycle peak column alternatives, FLOPs per unit)
COMPUTE_CEILINGS = {
    "FP64 FMA": (["sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained"], 2),
    "FP32 FMA": (["sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"], 2),
    "FP16 Tensor": (["sm__ops_path_tensor_src_fp16_dst_fp32.sum.peak_sustained",
                     "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.peak_sustained"], 1),
}
# VQE never touches tensor cores (see stage4_compute_opt/kernels/gate_apply_tc.cu);
# GQE's stage3/4 linear layers are fp16 GEMMs.
WORKLOAD_COMPUTE_ROOFS = {
    "vqe": ["FP64 FMA", "FP32 FMA"],
    "gqe": ["FP64 FMA", "FP32 FMA", "FP16 Tensor"],
}

# The report each stage's profile.sh writes for each workload, in tutorial order.
COMPARE_REPORTS = {
    "vqe": [
        ("Stage 1", "stage1_naive/profiles/stage1_vqe_roofline"),
        ("Stage 2", "stage2_gpu_native/profiles/stage2_vqe_fused_roofline"),
        ("Stage 3 (fp64)", "stage3_memory_opt/profiles/stage3_vqe_fp64_roofline"),
        ("Stage 3 (fp32)", "stage3_memory_opt/profiles/stage3_vqe_fp32_roofline"),
        ("Stage 4", "stage4_compute_opt/profiles/stage4_vqe_roofline"),
    ],
    "gqe": [
        ("Stage 1", "stage1_naive/profiles/stage1_gqe_roofline"),
        ("Stage 2", "stage2_gpu_native/profiles/stage2_gqe_roofline"),
        ("Stage 3 (fp16)", "stage3_memory_opt/profiles/stage3_gqe_fp16_roofline"),
        ("Stage 4 (tensor core)", "stage4_compute_opt/profiles/stage4_gqe_tc_roofline"),
    ],
}

# --- plot style -------------------------------------------------------------------

LEVEL_MARKERS = {"L1": "o", "L2": "s", "HBM": "v"}
LEVEL_COLORS = {"L1": "b", "L2": "r", "HBM": "g"}
KERNEL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                 "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
ROOF_COLOR = "gray"
FIGSIZE = (10.67, 6.6)
MARKERSIZE = 12
# Filled markers with a thin dark outline, so overlapping points stay distinguishable.
MARKER_EDGE = dict(markeredgecolor="k", markeredgewidth=0.8)
MAXCHAR = 32


# --- parsing -------------------------------------------------------------------

_UNIT_PREFIX = {"": 1.0, "n": 1e-9, "u": 1e-6, "m": 1e-3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
_UNIT_BASES = ("byte", "second", "cycle", "inst", "hz", "s")


def _unit_scale(unit: str) -> float:
    """Factor converting a value printed in `unit` to base units. ncu
    auto-scales each column (e.g. "us", "Ghz", "Gbyte/s", "Kbyte/cycle") and
    writes the unit in the row after the header."""
    scale = 1.0
    for i, part in enumerate(unit.strip().split("/")):
        m = re.fullmatch(r"([numKMGT]?)(%s)" % "|".join(_UNIT_BASES), part)
        if not m:
            continue  # unitless counter, "%", etc.
        f = _UNIT_PREFIX[m.group(1)]
        scale = scale * f if i == 0 else scale / f
    return scale


def _to_float(s: str) -> float:
    s = (s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def export_csv(report: str) -> str:
    """Export a .ncu-rep to CSV next to it (`ncu --import ... --csv --page raw`)."""
    csv_path = report[: -len(".ncu-rep")] + ".csv"
    with open(csv_path, "w") as f:
        subprocess.run(["ncu", "--import", report, "--csv", "--page", "raw"], stdout=f, check=True)
    return csv_path


def load_report(csv_path: str, include_half: bool = False) -> tuple[dict[str, dict], dict[str, float], dict[str, float]]:
    """Sum FLOPs, bytes and time over every launch of each kernel. Returns
    ({kernel name: totals}, {level: ceiling GB/s}, {compute ceiling: GFLOP/s})."""
    with open(csv_path, newline="") as f:
        rows = list(csv_module.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{csv_path}: empty CSV")
    header, units = rows[0], rows[1]
    col = {name: i for i, name in enumerate(header)}
    for needed in ("Kernel Name", TIME_METRIC, SMSP_CLOCK):
        if needed not in col:
            raise ValueError(f"{csv_path}: no '{needed}' column -- is this an "
                             "`ncu --set roofline` report exported with `--csv --page raw`?")
    scale = {name: _unit_scale(units[i]) for name, i in col.items()}
    data = [r for r in rows[2:] if len(r) == len(header) and r[col["Kernel Name"]]]

    def get(row, metric):
        return _to_float(row[col[metric]]) * scale[metric]

    def first(alternatives):
        return next((a for a in alternatives if a[0] in col), None)

    traffic = {lv: first(TRAFFIC_METRICS[lv]) for lv in LEVELS}
    missing = [lv for lv, t in traffic.items() if t is None]
    if missing:
        raise ValueError(f"{csv_path}: no {'/'.join(missing)} traffic metrics -- profile with `--set roofline`")
    flop_metrics = CC_FLOP_METRICS + (HALF_FLOP_METRICS if include_half else [])
    flop_cols = [a for a in (first(alts) for alts in flop_metrics) if a]
    tc_cols = []
    for m in TC_FLOP_METRICS:
        tc_cols += [m + ".sum"] if m + ".sum" in col else \
            [c for c in (f"{m}_sparsity_off.sum", f"{m}_sparsity_on.sum") if c in col]

    kernels: dict[str, dict] = {}
    for row in data:
        name = row[col["Kernel Name"]]
        k = kernels.setdefault(name, {"count": 0, "time": 0.0, "cc_flops": 0.0, "tc_flops": 0.0,
                                      **{lv: 0.0 for lv in LEVELS}})
        t = get(row, TIME_METRIC)
        k["count"] += 1
        k["time"] += t
        cycles = get(row, SMSP_CLOCK) * t
        k["cc_flops"] += sum(get(row, c) * w for c, w in flop_cols) * cycles
        k["tc_flops"] += sum(get(row, c) for c in tc_cols)
        for lv, (rate, width, _, _) in traffic.items():
            k[lv] += get(row, rate) * width * t

    for k in kernels.values():
        k["flops"] = k["cc_flops"] + k["tc_flops"]
        k["gflops"] = k["flops"] / k["time"] / 1e9 if k["time"] > 0 else 0.0
        for lv in LEVELS:
            k["AI " + lv] = k["flops"] / k[lv] if k[lv] > 0 else 0.0

    # Ceilings: per-cycle peak * clock. Clocks are locked during profiling,
    # so every launch gives the same value; the median guards against a stray row.
    def ceiling(per_cycle_col, clock_col, factor):
        vals = [get(r, per_cycle_col) * get(r, clock_col) * factor for r in data]
        return statistics.median(vals) if vals else 0.0

    mem_roofs = {lv: ceiling(peak, clock, width) / 1e9 for lv, (_, width, peak, clock) in traffic.items()}
    comp_roofs = {}
    for name, (alts, factor) in COMPUTE_CEILINGS.items():
        peak = next((c for c in alts if c in col), None)
        if peak and SM_CLOCK in col:
            comp_roofs[name] = ceiling(peak, SM_CLOCK, factor) / 1e9
    return kernels, mem_roofs, comp_roofs


def aggregate(kernels: dict[str, dict]) -> dict:
    """Whole-workload totals: one point per level for the entire report."""
    tot = {key: sum(k[key] for k in kernels.values()) for key in ["time", "flops"] + LEVELS}
    tot["gflops"] = tot["flops"] / tot["time"] / 1e9 if tot["time"] > 0 else 0.0
    for lv in LEVELS:
        tot["AI " + lv] = tot["flops"] / tot[lv] if tot[lv] > 0 else 0.0
    return tot


def short_name(name: str) -> str:
    """'void cub::Foo<...>(int, double *)' -> 'Foo', truncated to MAXCHAR."""
    s = re.sub(r"^void\s+", "", name)
    s = s.split("(")[0].split("<")[0].split("::")[-1].strip() or name
    return s if len(s) <= MAXCHAR else s[: MAXCHAR - 1] + "…"


# --- ceilings ----------------------------------------------------------------


def select_roofs(workload: str, mem_roofs: dict, comp_roofs: dict, override_path: str | None) -> tuple[dict, dict]:
    """Restrict compute ceilings to the workload's and apply --roofs overrides."""
    mem, comp = dict(mem_roofs), dict(comp_roofs)
    if override_path:
        with open(override_path) as f:
            override = json.load(f)
        mem.update(override.get("mem", {}))
        comp.update(override.get("compute", {}))
    return mem, {n: comp[n] for n in WORKLOAD_COMPUTE_ROOFS[workload] if comp.get(n, 0) > 0}


def stage0_hbm() -> float | None:
    """stage0's measured HBM bandwidth (GB/s), drawn as a dashed reference line."""
    try:
        with open(os.path.join(REPO_ROOT, "stage0_calibration", "bandwidth_fp64.json")) as f:
            return float(json.load(f)["achieved_gbps"])
    except (OSError, KeyError, ValueError):
        return None


# --- plotting -------------------------------------------------------------------


def _axis_limits(points: list[tuple[float, float]], comp_roofs: dict) -> tuple[float, float, float, float]:
    """[1e-3, 1e3] x [1, 5x top ceiling], widened by whole decades so every
    point is visible."""
    xmin, xmax, ymin = 1e-3, 1e3, 1.0
    ymax = max(comp_roofs.values()) * 5
    for ai, gf in points:
        xmin = min(xmin, 10 ** np.floor(np.log10(ai)))
        xmax = max(xmax, 10 ** np.ceil(np.log10(ai)))
        ymin = min(ymin, 10 ** np.floor(np.log10(gf)))
    return xmin, xmax, ymin, ymax


def _bw_label(ax, text, bw, y_label, bold=True, ha="right"):
    """Bandwidth label rotated to lie along its (log-log) ceiling line, ending
    (ha="right") or starting (ha="left") at y = y_label."""
    x_label = y_label / bw
    p0, p1 = ax.transData.transform([(x_label / 10, y_label / 10), (x_label, y_label)])
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.annotate(text, xy=(x_label, y_label), xytext=(3, -5), textcoords="offset points", color=ROOF_COLOR,
                rotation=ang, rotation_mode="anchor", horizontalalignment=ha, verticalalignment="top",
                weight="bold" if bold else "normal", fontsize=10)


def plot_roofs(fig, ax, limits, mem_roofs: dict, comp_roofs: dict, levels: list[str]):
    """Gray ceilings with bold annotations."""
    xmin, xmax, ymin, ymax = limits
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]")
    ax.set_ylabel("Performance [GFLOP/s]")

    x = np.logspace(np.log10(xmin), np.log10(xmax), 2000)
    for lv in levels:
        for comp in comp_roofs.values():
            ax.plot(x, np.minimum(mem_roofs[lv] * x, comp), c=ROOF_COLOR, ls="-", lw=2)
    measured = stage0_hbm() if "HBM" in levels else None
    if measured:
        ax.plot(x, np.minimum(measured * x, max(comp_roofs.values())), c=ROOF_COLOR, ls="--", lw=1.5)
    for name, comp in comp_roofs.items():
        ax.annotate(f"{name}: {comp / 1000:.1f} TFLOP/s", xy=(xmax, comp), xytext=(-5, 5),
                    textcoords="offset points", weight="bold", color=ROOF_COLOR,
                    horizontalalignment="right", fontsize=10)

    # Rotated bandwidth labels need the final axes geometry to get the angle right.
    fig.tight_layout()
    y_label = min(1000.0, min(comp_roofs.values()) / 10)
    for lv in levels:
        _bw_label(ax, f"{lv}: {mem_roofs[lv]:.1f} GB/s", mem_roofs[lv], y_label)
    if measured:
        # Start at the left edge, where the region under the HBM ceiling is empty.
        x_start = max(xmin * 1.5, ymin * 2 / measured)
        _bw_label(ax, f"stage0 measured HBM: {measured:.1f} GB/s", measured, measured * x_start, bold=False, ha="left")


def _title(ax, text):
    ax.text(0.01, 0.99, text, transform=ax.transAxes, ha="left", va="top", fontsize=13)


def _level_legend(ax, levels):
    handles = [ax.plot([], [], c="k", marker=LEVEL_MARKERS[lv], linestyle="None", ms=10,
                       markerfacecolor="gray", **MARKER_EDGE, label=lv)[0] for lv in levels]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 0.93), ncol=len(levels),
              fontsize=11, frameon=False)


def plot_kernels(path, title, kernels: list[tuple[str, dict]], levels, mem_roofs, comp_roofs):
    """One color per kernel; one marker per memory level (circles for single-level plots)."""
    pts = [(k["AI " + lv], k["gflops"]) for _, k in kernels for lv in levels]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_roofs(fig, ax, _axis_limits(pts, comp_roofs), mem_roofs, comp_roofs, levels)

    single_level = len(levels) == 1
    for i, (label, k) in enumerate(kernels):
        c = KERNEL_COLORS[i % len(KERNEL_COLORS)]
        for lv in levels:
            ax.plot(k["AI " + lv], k["gflops"], c=c, marker="o" if single_level else LEVEL_MARKERS[lv], linestyle="None",
                    ms=MARKERSIZE, **MARKER_EDGE)

    kernel_handles = [mpatches.Patch(color=KERNEL_COLORS[i % len(KERNEL_COLORS)], label=label)
                      for i, (label, _) in enumerate(kernels)]
    leg = ax.legend(handles=kernel_handles, loc="lower right", fontsize=9, frameon=False)
    if not single_level:
        ax.add_artist(leg)
        _level_legend(ax, levels)
    _title(ax, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overview(path, title, groups: list[tuple[str, dict]], levels, mem_roofs, comp_roofs):
    """Aggregate L1/L2/HBM points. One group -> blue/red/green dots; several
    groups (--compare) -> one color per stage, marker per level."""
    pts = [(g["AI " + lv], g["gflops"]) for _, g in groups for lv in levels if g["AI " + lv] > 0]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_roofs(fig, ax, _axis_limits(pts, comp_roofs), mem_roofs, comp_roofs, levels)

    if len(groups) == 1:
        g = groups[0][1]
        for lv in levels:
            ax.plot(g["AI " + lv], g["gflops"], c=LEVEL_COLORS[lv], marker="o", ms=15, linestyle="None", **MARKER_EDGE)
        ax.legend(handles=[mpatches.Patch(color=LEVEL_COLORS[lv], label=lv) for lv in levels],
                  loc="lower right", frameon=False)
    else:
        for i, (label, g) in enumerate(groups):
            c = KERNEL_COLORS[i % len(KERNEL_COLORS)]
            for lv in levels:
                if g["AI " + lv] > 0:
                    ax.plot(g["AI " + lv], g["gflops"], c=c, marker=LEVEL_MARKERS[lv], linestyle="None",
                            ms=MARKERSIZE, **MARKER_EDGE)
        leg = ax.legend(handles=[mpatches.Patch(color=KERNEL_COLORS[i % len(KERNEL_COLORS)], label=label)
                                 for i, (label, _) in enumerate(groups)],
                        loc="lower right", fontsize=11, frameon=False)
        ax.add_artist(leg)
        _level_legend(ax, levels)
    _title(ax, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- drivers --------------------------------------------------------------------


def _resolve_csv(report: str) -> str:
    """Accept '<x>.ncu-rep', '<x>.csv' or the bare '<x>'; export the CSV if
    it's missing or older than the report."""
    base = re.sub(r"\.(ncu-rep|csv)$", "", report)
    rep, csv_path = base + ".ncu-rep", base + ".csv"
    if os.path.exists(rep) and (not os.path.exists(csv_path) or os.path.getmtime(csv_path) < os.path.getmtime(rep)):
        print(f"Exporting {rep} -> {csv_path}")
        return export_csv(rep)
    if os.path.exists(csv_path):
        return csv_path
    raise FileNotFoundError(f"neither {rep} nor {csv_path} exists")


def _workload(path: str) -> str:
    return "gqe" if "gqe" in os.path.basename(path).lower() else "vqe"


def write_kernel_table(path: str, kernels: dict[str, dict]):
    keys = ["count", "time", "cc_flops", "tc_flops", "flops", "L1", "L2", "HBM",
            "AI L1", "AI L2", "AI HBM", "gflops"]
    with open(path, "w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["Kernel Name"] + keys)
        for name, k in sorted(kernels.items(), key=lambda kv: -kv[1]["time"]):
            w.writerow([name] + [k[key] for key in keys])


def plot_report(report: str, args) -> None:
    csv_path = _resolve_csv(report)
    tag = os.path.basename(csv_path)[: -len(".csv")]
    out_dir = args.out_dir or os.path.join(os.path.dirname(csv_path), "roofline_plots")
    os.makedirs(out_dir, exist_ok=True)

    kernels, mem_roofs, comp_roofs = load_report(csv_path, args.include_half)
    mem_roofs, comp_roofs = select_roofs(args.workload or _workload(csv_path), mem_roofs, comp_roofs, args.roofs)
    write_kernel_table(os.path.join(out_dir, f"{tag}_kernels.csv"), kernels)

    by_time = sorted(kernels.items(), key=lambda kv: -kv[1]["time"])
    plotted = [(n, k) for n, k in by_time
               if k["gflops"] > 0 and all(k["AI " + lv] > 0 for lv in LEVELS)][: args.top]
    skipped = len([1 for _, k in by_time if k["flops"] == 0])

    print(f"\n{tag}: {len(kernels)} distinct kernels ({skipped} with zero FLOPs, e.g. copies/fills, "
          f"can't go on a log-scale roofline); plotting the top {len(plotted)} by GPU time")
    print("  ceilings: " + ", ".join(f"{lv} {mem_roofs[lv]:.0f} GB/s" for lv in LEVELS) + ", "
          + ", ".join(f"{n} {v / 1000:.1f} TFLOP/s" for n, v in comp_roofs.items()))
    print(f"  {'kernel':<{MAXCHAR}}  {'calls':>5}  {'time(us)':>10}  {'GFLOP/s':>9}  "
          + "  ".join(f"{'AI ' + lv:>8}" for lv in LEVELS))
    for name, k in plotted:
        print(f"  {short_name(name):<{MAXCHAR}}  {k['count']:>5}  {k['time'] * 1e6:>10.1f}  {k['gflops']:>9.2f}  "
              + "  ".join(f"{k['AI ' + lv]:>8.3f}" for lv in LEVELS))
    if not plotted:
        print(f"WARNING: {tag}: no kernel has nonzero FLOPs and bytes at every level; nothing to plot.")
        return

    labels, seen = [], {}
    for name, _ in plotted:  # keep labels unique after shortening
        s = short_name(name)
        seen[s] = seen.get(s, 0) + 1
        labels.append(s if seen[s] == 1 else f"{s} [{seen[s]}]")
    labeled = list(zip(labels, [k for _, k in plotted]))

    written = [os.path.join(out_dir, f"{tag}_all.png")]
    plot_kernels(written[-1], tag, labeled, LEVELS, mem_roofs, comp_roofs)
    for lv in LEVELS:
        written.append(os.path.join(out_dir, f"{tag}_{lv.lower()}.png"))
        plot_kernels(written[-1], f"{tag} ({lv})", labeled, [lv], mem_roofs, comp_roofs)
    tot = aggregate(kernels)
    if tot["gflops"] > 0:
        written.append(os.path.join(out_dir, f"{tag}_overview.png"))
        plot_overview(written[-1], f"{tag}: all kernels", [(tag, tot)], LEVELS, mem_roofs, comp_roofs)
    print(f"  overview: {tot['gflops']:.2f} GFLOP/s, "
          + ", ".join(f"AI {lv} {tot['AI ' + lv]:.3f}" for lv in LEVELS))
    print("  wrote " + ", ".join(written))


def plot_compare(workload: str, args) -> None:
    groups, mem_roofs, comp_roofs = [], None, None
    for label, base in COMPARE_REPORTS[workload]:
        try:
            csv_path = _resolve_csv(os.path.join(REPO_ROOT, base))
        except FileNotFoundError:
            print(f"  ({label}: no report yet at {base}.ncu-rep -- skipped)")
            continue
        kernels, mem, comp = load_report(csv_path, args.include_half)
        mem_roofs, comp_roofs = mem_roofs or mem, comp_roofs or comp
        tot = aggregate(kernels)
        if tot["gflops"] > 0:
            groups.append((label, tot))
    if not groups:
        print(f"No {workload.upper()} reports found for --compare; run the stages' profile.sh first.")
        return
    mem_roofs, comp_roofs = select_roofs(workload, mem_roofs, comp_roofs, args.roofs)
    out = args.out or os.path.join(REPO_ROOT, f"roofline_hierarchical_{workload}.png")
    plot_overview(out, f"{workload.upper()}: stage progression (all kernels)", groups, LEVELS, mem_roofs, comp_roofs)
    print(f"Wrote {out} ({', '.join(g[0] for g in groups)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="*", help=".ncu-rep or `ncu --csv --page raw` .csv files")
    ap.add_argument("--compare", choices=sorted(COMPARE_REPORTS), action="append", default=[],
                    help="Plot every stage's aggregate point for this workload on one chart (repeatable).")
    ap.add_argument("--workload", choices=sorted(WORKLOAD_COMPUTE_ROOFS), default=None,
                    help="Which compute ceilings to draw (default: 'gqe' if the filename contains gqe, else 'vqe').")
    ap.add_argument("--roofs", default=None,
                    help='JSON overriding ncu\'s theoretical ceilings: {"mem": {"HBM": ...GB/s}, '
                         '"compute": {"FP64 FMA": ...GFLOP/s}}.')
    ap.add_argument("--include-half", action="store_true",
                    help="Also count CUDA-core fp16 (HADD/HMUL/HFMA) instructions as FLOPs. Off by default: "
                         "in this tutorial they are only compiler register moves (see module docstring).")
    ap.add_argument("--top", type=int, default=8, help="Max kernels per plot, by GPU time (default 8).")
    ap.add_argument("--out-dir", default=None, help="Per-report output directory (default: <report dir>/roofline_plots).")
    ap.add_argument("--out", default=None, help="Output path for --compare (default: roofline_hierarchical_<workload>.png).")
    args = ap.parse_args()
    plt.rc("font", size=15)

    if not args.reports and not args.compare:
        ap.error("give one or more reports, or --compare vqe|gqe")

    failed = False
    for report in args.reports:
        try:
            plot_report(report, args)
        except (OSError, ValueError, subprocess.CalledProcessError) as e:
            print(f"ERROR: {report}: {e}")
            failed = True
    for workload in args.compare:
        plot_compare(workload, args)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
