# Method

How every number in this repository was produced, and what each one does and
does not support. If a figure in the README cannot be traced back to something
described here, treat it as a bug.

## Order of work

The sequence matters, and it is deliberately not "write a fast kernel, then
find a benchmark that flatters it":

1. Benchmark the unmodified model end-to-end. Record tokens/sec and peak memory.
2. Profile it (`scripts/profile_nsys.sh`). Read the kernel summary. Identify
   where time actually goes.
3. Only then pick a target and write a kernel.
4. Prove the kernel is numerically correct (`make test`) before timing it once.
5. Re-benchmark, re-profile, and record both numbers.

Step 4 is not a formality. A kernel that skips the fp32 reduction in RMSNorm is
meaningfully faster and silently wrong for hidden sizes above ~2048, and the
error only shows up as slightly worse generations, not as a crash.

## What "percent of peak" means here

Two denominators are reported, because they answer different questions.

**Percent of datasheet.** The vendor's published bandwidth figure. Useful for
comparing across hardware, useless as a target, since no real kernel reaches
it.

**Percent of measured copy bandwidth.** A large device-to-device copy, sized
past any last-level cache, timed on the same machine in the same process
(`hag.microbench`). This is the honest ceiling: it is what the memory system
actually delivers to a kernel that does nothing but stream. A memory-bound
kernel at 90% of this figure has essentially no headroom left, and chasing the
remaining 10% is wasted effort.

The numerator is **ideal traffic, not measured traffic**: the minimum number of
bytes the operation must move given its inputs and outputs. For fused
residual-add + RMSNorm that is read `x`, read `residual`, write `h`, write `y`:
four arrays. Reporting against ideal traffic means the metric answers "how
close is this to the best possible implementation", which is the question worth
asking. It also means a kernel cannot flatter itself by moving extra bytes
quickly.

## Timing

Three details, each of which will silently inflate a speedup if ignored.

**Synchronisation.** CUDA and Metal launches are asynchronous. Timing without a
device sync before stopping the clock measures the submission, not the work.

**Cache flushing between replicates.** Every kernel here is memory-bound. Left
alone, the second replicate finds its input still in L2 and runs several times
faster than the first. `triton.testing.do_bench` handles this on CUDA and is
used wherever it is available; the fallback path in `hag.timing` dirties a
64 MiB buffer between replicates by hand.

**The launch floor.** A single decode row is a few kilobytes. The kernel
finishes long before a command buffer can be submitted and waited on. On the
M3 measured here that floor is around 175 microseconds, which is larger than
every decode-shaped measurement in the sweep.

That last point is why the results tables exclude decode-regime rows rather
than printing them. Those measurements are real, but they describe the
runtime's submission path, not the kernel, and presenting them as kernel
benchmarks would be misleading. They are kept in the JSON with a
`dispatch_bound` flag.

To actually resolve decode-regime kernel cost you need one of:

- an end-to-end benchmark, where the kernel runs inside a real forward pass and
  the launches are already amortised across layers (`hag.bench_e2e`);
- GPU counters via `ncu` on CUDA, or Xcode Instruments' Metal System Trace on
  Apple silicon, which time the kernel on the device rather than on the host;
- CUDA graphs, which remove per-launch submission cost entirely. That is the
  right fix in production, and out of scope here.

## Prefill and decode are different machines

They are reported separately throughout because they behave nothing alike:

| | prefill | decode |
| --- | --- | --- |
| shape | hundreds to thousands of rows | one row per sequence |
| bound by | compute (GEMM) | memory bandwidth |
| what fusion buys | fewer bytes through DRAM | fewer launches |

A kernel that wins in one and not the other is the normal result. Averaging
them into a single "speedup" hides the only interesting part.

## Numerical tolerance

Kernels are checked against a PyTorch/MLX reference, not against an exact
answer. Both sides reduce in fp32 and store in the working dtype, so agreement
should be within one rounding of the final store: `2e-2` for fp16, `1e-5` for
fp32. Tolerances that had to be loosened to make a test pass are a signal the
kernel is wrong, not that the test is strict.

The saturating-input test exists because it is the specific case where a
plausible-looking fp16 sigmoid diverges from the reference while every
randomly-sampled test still passes.

## Known limits

- Forward pass only. There is no backward; this is an inference project.
- The Triton kernels are unverified on hardware until the CUDA half of the test
  suite runs green on an actual NVIDIA device. Until then they are code, not
  results.
- The end-to-end patcher replaces RMSNorm and the MLP activation. It does not
  touch attention, so end-to-end gains are bounded by the fraction of the
  forward pass those two occupy, which is exactly what the profile in step 2
  is for.
- MLX has no end-to-end path here yet; the Apple-silicon numbers are op-level
  plus an unpatched baseline.
