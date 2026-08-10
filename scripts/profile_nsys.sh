#!/usr/bin/env bash
# Capture an Nsight Systems timeline of the end-to-end benchmark.
#
# This is step 2 of the project: find the traffic jam before writing a kernel.
# Run it BEFORE touching any kernel code, keep the report, and run it again
# after. The pair of traces is the evidence; the kernel is just the fix.
#
#   ./scripts/profile_nsys.sh Qwen/Qwen2.5-1.5B
#
# nsys works fine inside Colab and Kaggle notebooks via `!`. It is `ncu`
# (scripts/profile_ncu.sh) that needs a VM with performance-counter access.
# Download the .nsys-rep and open it in the Nsight Systems GUI, which has a
# macOS host build.

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B}"
TAG="${2:-baseline}"
OUT_DIR="profiles"
mkdir -p "$OUT_DIR"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found. On a Colab/Kaggle runtime:" >&2
    echo "  apt-get -qq install -y nsight-systems-cli" >&2
    echo "On a GCP/Azure Deep Learning VM it ships with the CUDA toolkit." >&2
    exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${OUT_DIR}/${TAG}_${STAMP}"

# --trace: CUDA API + kernels + NVTX ranges + OS runtime, which is what shows
#   the gaps between kernels rather than just the kernels themselves.
# --cuda-memory-usage: annotates the timeline with allocation traffic, the
#   thing this project is actually hunting.
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="$REPORT" \
    python -m hag.bench_e2e --model "$MODEL" --prompt-tokens 512 --new-tokens 64

echo
echo "Timeline written to ${REPORT}.nsys-rep"
echo
echo "Text summaries without opening the GUI:"
echo "  nsys stats --report cuda_gpu_kern_sum ${REPORT}.nsys-rep   # where time goes"
echo "  nsys stats --report cuda_gpu_mem_time_sum ${REPORT}.nsys-rep  # transfer time"
echo
echo "Read the kernel summary first. If the top entries are elementwise ops"
echo "rather than GEMMs, the model is memory-bound and fusion is the lever."
