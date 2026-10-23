#!/usr/bin/env python3
"""Cross-check plot_hierarchical_roofline.py against the values the Nsight
Compute GUI draws in its "GPU Speed Of Light Hierarchical Roofline" charts.

The GUI values are computed here per *launch* and per precision chart
(Double / Single / Half / Tensor), straight from the section-file
expressions, using ncu's own derived__* columns where the section defines
them (so this does not share arithmetic with plot_hierarchical_roofline.py).

The GUI's Half chart is excluded from the comparison, matching
plot_hierarchical_roofline.py's default (see its docstring: in this
tutorial those counts are compiler register moves, not math).

Usage: python3 tools/verify_vs_gui.py <report.csv> [...]
(tools/run_roofline_analysis.sh runs this over every stage's reports.)
"""
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_hierarchical_roofline as phr  # noqa: E402

PC = "smsp__sass_thread_inst_executed_op_{}_pred_on.sum.per_cycle_elapsed"
DER = "derived__smsp__sass_thread_inst_executed_op_{}"
CHARTS = {  # GUI chart -> per-cycle achieved-work columns (already weighted), clock
    "Double": ([PC.format("dadd"), PC.format("dmul"), DER.format("dfma_pred_on_x2")], "smsp__cycles_elapsed.avg.per_second"),
    "Single": ([PC.format("fadd"), PC.format("fmul"), DER.format("ffma_pred_on_x2")], "smsp__cycles_elapsed.avg.per_second"),
    "Half": ([DER.format("hadd_pred_on_x2"), DER.format("hmul_pred_on_x2"), DER.format("hfma_pred_on_x4")],
             "smsp__cycles_elapsed.avg.per_second"),
    "Tensor": ([f"sm__ops_path_tensor_src_{s}.sum.per_cycle_elapsed" for s in
                ("fp64", "tf32_dst_fp32_sparsity_off", "fp16_dst_fp16_sparsity_off",
                 "fp16_dst_fp32_sparsity_off", "bf16_dst_fp32_sparsity_off")], "sm__cycles_elapsed.avg.per_second"),
}
TRAFFIC = {"L1": "derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second",
           "L2": "derived__lts__lts2xbar_bytes.sum.per_second",
           "HBM": "dram__bytes.sum.per_second"}
CEIL_MEM = {"L1": ("derived__l1tex__lsu_writeback_bytes_mem_lg.sum.peak_sustained", "l1tex__cycles_elapsed.avg.per_second"),
            "L2": ("derived__lts__lts2xbar_bytes.sum.peak_sustained", "lts__cycles_elapsed.avg.per_second"),
            "HBM": ("dram__bytes.sum.peak_sustained", "dram__cycles_elapsed.avg.per_second")}
CEIL_COMP = {"FP64 FMA": "derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2",
             "FP32 FMA": "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
             "FP16 Tensor": "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.peak_sustained"}


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def gui_launches(path):
    rows = list(csv.reader(open(path, newline="")))
    h, u = rows[0], rows[1]
    col = {n: i for i, n in enumerate(h)}
    sc = {n: phr._unit_scale(u[i]) for n, i in col.items()}
    g = lambda r, m: phr._to_float(r[col[m]]) * sc[m] if m in col else 0.0
    data = [r for r in rows[2:] if len(r) == len(h) and r[col["Kernel Name"]]]
    launches = []
    for r in data:
        work = {c: sum(g(r, m) for m in ms) * g(r, clk) for c, (ms, clk) in CHARTS.items()}
        traffic = {lv: g(r, m) for lv, m in TRAFFIC.items()}
        launches.append((r[col["Kernel Name"]], work, traffic, g(r, "gpu__time_duration.sum")))
    sm_clk = "sm__cycles_elapsed.avg.per_second"
    ceil = {lv: statistics.median(g(r, p) * g(r, c) for r in data) / 1e9 for lv, (p, c) in CEIL_MEM.items()}
    ceil.update({n: statistics.median(g(r, m) * g(r, sm_clk) for r in data) / 1e9 for n, m in CEIL_COMP.items()})
    return launches, ceil


def main():
    worst = 0.0
    for path in sys.argv[1:]:
        launches, gui_ceil = gui_launches(path)
        kernels, mem, comp = phr.load_report(path)
        mine_ceil = {**mem, **comp}
        print(f"\n=== {os.path.basename(path)}: {len(launches)} launches, {len(kernels)} kernels")
        cerr = max(rel(gui_ceil[k], mine_ceil[k]) for k in gui_ceil if k in mine_ceil)
        print("  ceilings GUI : " + ", ".join(f"{k} {v:.1f}" for k, v in gui_ceil.items()))
        print(f"  ceilings max rel diff vs PNG: {cerr:.2e}")
        worst = max(worst, cerr)

        # Per kernel: aggregate GUI per-launch values the same way (time-weighted)
        # requires durations; single-launch kernels compare 1:1 with no weighting.
        n_cmp, kerr = 0, 0.0
        for name, k in kernels.items():
            ls = [l for l in launches if l[0] == name]
            if not ls or k["gflops"] <= 0:
                continue
            # Several launches: the PNG point is total FLOPs / total time, i.e. the
            # time-weighted mean of the GUI's per-launch rates.
            T = sum(l[3] for l in ls)
            work = {c: sum(l[1][c] * l[3] for l in ls) / T for c in CHARTS}
            traffic = {lv: sum(l[2][lv] * l[3] for l in ls) / T for lv in phr.LEVELS}
            plotted = sum(v for c, v in work.items() if c != "Half")
            gf = plotted / 1e9
            errs = [rel(gf, k["gflops"])]
            for lv in phr.LEVELS:
                if traffic[lv] > 0 and k["AI " + lv] > 0:
                    errs.append(rel(plotted / traffic[lv], k["AI " + lv]))
            n_cmp += 1
            e = max(errs)
            kerr = max(kerr, e)
            prec = ", ".join(f"{c} {v / 1e9:.3g}" for c, v in work.items() if v > 0)
            print(f"  {phr.short_name(name):<32} x{len(ls):<3} GUI GFLOP/s by chart: [{prec}]  "
                  f"PNG {k['gflops']:.4g} GFLOP/s, AI L1/L2/HBM "
                  + "/".join(f"{k['AI ' + lv]:.3g}" for lv in phr.LEVELS) + f"  max rel diff {e:.1e}")
        print(f"  kernels compared (all launches): {n_cmp}, max rel diff {kerr:.2e}")
        worst = max(worst, kerr)
    print(f"\nOVERALL max relative difference PNG vs GUI: {worst:.2e}")


if __name__ == "__main__":
    main()
