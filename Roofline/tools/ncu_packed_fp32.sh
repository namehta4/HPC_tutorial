# Sourced by every stage's profile.sh; sets NCU_PACKED_FP32 to extra ncu arguments.
#
# On compute capability 10.x (B200, GB200, B300, GB300), cuBLAS's SIMT SGEMM
# kernels issue packed FP32 instructions (FFMA2/FADD2/FMUL2).
# plot_hierarchical_roofline.py already counts them, but ncu 2025.3's
# `--set roofline` doesn't collect their counters, so without these the FP32
# work of those kernels is missing from the roofline. Other GPUs have no
# packed FP32 counters, so nothing is added there.
NCU_PACKED_FP32=()
_gpu="${CUDA_VISIBLE_DEVICES:-0}"; _gpu="${_gpu%%,*}"; _gpu="${_gpu:-0}"
_cc="$(nvidia-smi -i "$_gpu" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)" || _cc=
if [ "${_cc%%.*}" = 10 ]; then
    NCU_PACKED_FP32=(--metrics "$(printf 'smsp__sass_thread_inst_executed_op_%s_pred_on.sum.per_cycle_elapsed,' ffma2 fadd2 fmul2 | sed 's/,$//')")
fi
unset _gpu _cc
