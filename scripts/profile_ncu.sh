#!/usr/bin/env bash
# Per-kernel memory counters with Nsight Compute.
#
# This is the tool that turns "my kernel got faster" into "my kernel moves N
# bytes and sustains X% of peak DRAM throughput". It needs GPU
# performance-counter access, which is the reason a plain Colab runtime is not
# enough:
#
#   ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU
#   Performance Counters on the target device.
#
# Fix on a VM you control (GCP/Azure/Lambda), then reboot:
#   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
#     | sudo tee /etc/modprobe.d/nvidia-profiler.conf
#   sudo update-initramfs -u && sudo reboot
#
# Or just run it under sudo, which is usually simpler on a throwaway box.

set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not found. Install the Nsight Compute CLI:" >&2
    echo "  apt-get -qq install -y nsight-compute" >&2
    exit 1
fi

OUT_DIR="profiles"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Only the custom kernels are profiled: ncu serialises and replays every kernel
# it touches, so profiling a whole forward pass takes minutes and tells you
# little you did not already get from nsys.
ncu \
    --kernel-name-base demangled \
    --kernel-name 'regex:(_swiglu_fwd|_rmsnorm)' \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    --section Occupancy \
    --export "${OUT_DIR}/kernels_${STAMP}" \
    --force-overwrite \
    python -m hag.bench_ops --backend cuda --dtype fp16 --warmup 2 --iters 3

echo
echo "Wrote ${OUT_DIR}/kernels_${STAMP}.ncu-rep"
echo
echo "The number that matters is 'Memory Throughput [%]' in Speed of Light."
echo "For a memory-bound kernel it should approach 100%. If it does not, the"
echo "kernel is leaving bandwidth on the table -- check for uncoalesced access"
echo "in Memory Workload Analysis before blaming occupancy."
