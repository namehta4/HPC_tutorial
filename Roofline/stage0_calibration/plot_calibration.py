#!/usr/bin/env python3
"""Stage 0 roofline: every ceiling of this GPU, with no kernels on it yet.

Two sets of ceilings are drawn on one chart:
  * theoretical (solid gray): L1, L2, HBM bandwidth and FP64 FMA, FP32 FMA,
    FP16 tensor-core peaks, read from an existing ncu `--set roofline` CSV
    with plot_hierarchical_roofline.py's own parser, so they are exactly the
    ceilings the per-stage plots use (peak_sustained x the clocks ncu locked);
  * measured by stage0 (dashed black): STREAM Triad HBM bandwidth
    (bandwidth_fp64.json) and cuBLAS GEMM peaks on FP64 / TF32 / FP16 tensor
    cores (gemm_peak.json).
Stage0 can't measure L1/L2, so those exist only as theoretical ceilings.
Measured GEMM peaks can exceed ncu's ceilings because ncu locks clocks at
base frequency while the probes run at boost.

Usage:  python3 plot_calibration.py [--ncu-csv <report>.csv] [--out roofline_calibration.png]
Without --ncu-csv, the first stage*/profiles/*_roofline.csv found is used
(GQE reports first, since only they're guaranteed to carry the tensor peak);
with none at all, only the measured ceilings are drawn.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
import plot_hierarchical_roofline as phr  # noqa: E402

THEORY_COLOR = "gray"
MEASURED_COLOR = "k"
FIGSIZE = (10.67, 6.6)


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def find_ncu_csv():
    for pattern in ("stage*/profiles/*gqe*_roofline.csv", "stage*/profiles/*_roofline.csv"):
        hits = sorted(glob.glob(os.path.join(REPO_ROOT, pattern)))
        if hits:
            return hits[-1]
    return None


def bw_label(ax, text, bw, y_label, color, above, bold):
    """Label rotated along a (log-log) bandwidth line ending at y = y_label,
    placed above (left of) or below (right of) the line."""
    x_label = y_label / bw
    p0, p1 = ax.transData.transform([(x_label / 10, y_label / 10), (x_label, y_label)])
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.annotate(text, xy=(x_label, y_label), xytext=(-3, 4) if above else (3, -5), textcoords="offset points",
                color=color, rotation=ang, rotation_mode="anchor", ha="right",
                va="bottom" if above else "top", weight="bold" if bold else "normal", fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncu-csv", default=None, help="ncu --set roofline CSV to take theoretical ceilings from")
    ap.add_argument("--out", default=os.path.join(HERE, "roofline_calibration.png"))
    args = ap.parse_args()

    bw_json, gemm = load("bandwidth_fp64.json"), load("gemm_peak.json")
    device = bw_json.get("device", gemm.get("device", "GPU"))
    meas_bw = float(bw_json["achieved_gbps"])  # GB/s
    # name -> (GFLOP/s, label placement): "edge" = right-aligned at the plot's right edge,
    # "ridge" = starting at the ridge point; above/below the line. Chosen so the
    # A100's closely spaced ceilings (15.1 vs 18.4, 241 vs 272 TFLOP/s) don't collide.
    meas_comp = {"FP64 Tensor GEMM (measured)": (gemm["fp64"] * 1e3, "ridge", True),
                 "TF32 Tensor GEMM (measured)": (gemm["fp32_tf32"] * 1e3, "ridge", False),
                 "FP16 Tensor GEMM (measured)": (gemm["fp16"] * 1e3, "edge", True)}

    csv_path = args.ncu_csv or find_ncu_csv()
    theory_mem, theory_comp = {}, {}
    if csv_path:
        _, theory_mem, comp = phr.load_report(csv_path)
        theory_comp = {f"{n} (theoretical)": (v, "edge", False) for n, v in comp.items()}
        print(f"theoretical ceilings from {os.path.relpath(csv_path, REPO_ROOT)}")
    else:
        print("no ncu roofline CSV found: drawing measured ceilings only")

    all_comp = {**theory_comp, **meas_comp}
    top = max(v for v, _, _ in all_comp.values())
    xmin, xmax, ymin, ymax = 1e-2, 1e5, 10.0, top * 5
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    ax.set_title(f"Stage 0: roofline ceilings, {device}")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, color="gray", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="gray", alpha=0.3)

    x = np.logspace(np.log10(xmin), np.log10(xmax), 2000)
    for bw in theory_mem.values():
        for v, _, _ in theory_comp.values():
            ax.plot(x, np.minimum(bw * x, v), c=THEORY_COLOR, ls="-", lw=2)
    for v, _, _ in meas_comp.values():
        ax.plot(x, np.minimum(meas_bw * x, v), c=MEASURED_COLOR, ls="--", lw=1.5)

    for name, (v, where, above) in all_comp.items():
        measured = name in meas_comp
        bw = meas_bw if measured or "HBM" not in theory_mem else theory_mem["HBM"]
        text = f"{name}: {v / 1e3:.1f} TFLOP/s"
        if where == "edge":
            xy, ha, dx = (xmax, v), "right", -5
        else:
            xy, ha, dx = (v / bw, v), "left", 5
        ax.annotate(text, xy=xy, xytext=(dx, 4 if above else -4), textcoords="offset points",
                    va="bottom" if above else "top", ha=ha, fontsize=10,
                    color=MEASURED_COLOR if measured else THEORY_COLOR, weight="normal" if measured else "bold")

    # Rotated bandwidth labels need the final axes geometry to get the angle right.
    fig.tight_layout()
    for lv, bw in theory_mem.items():
        bw_label(ax, f"{lv} (theoretical): {bw:.1f} GB/s", bw, 3000.0, THEORY_COLOR, above=True, bold=True)
    bw_label(ax, f"HBM (measured, STREAM Triad): {meas_bw:.1f} GB/s", meas_bw, 3000.0, MEASURED_COLOR,
             above=False, bold=False)

    handles = [mlines.Line2D([], [], c=MEASURED_COLOR, ls="--", lw=1.5, label="measured by stage0")]
    if csv_path:
        handles.insert(0, mlines.Line2D([], [], c=THEORY_COLOR, ls="-", lw=2,
                                        label="theoretical (ncu, locked clocks)"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
