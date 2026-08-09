"""Measurement primitives.

Two details matter more than anything else in this file, and both are easy to
get wrong in a way that silently inflates a speedup:

1. **Synchronize before you stop the clock.** CUDA and Metal launches are
   asynchronous. Timing without a sync measures launch overhead.
2. **Flush the L2 between replicates.** Every kernel in this repo is
   memory-bound. If the input still sits in L2 from the previous replicate you
   are timing cache, not DRAM, and the number can be several times too good.
   `triton.testing.do_bench` does this for us on CUDA; the fallback path below
   does it by hand.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable


def _sync(backend: str) -> None:
    if backend == "cuda":
        import torch

        torch.cuda.synchronize()
    elif backend == "mps":
        import torch

        torch.mps.synchronize()
    elif backend == "mlx":
        import mlx.core as mx

        mx.synchronize()


def bench_ms(
    fn: Callable[[], object],
    backend: str,
    warmup: int = 25,
    iters: int = 100,
    flush_l2: bool = True,
    inner_reps: int = 1,
) -> dict[str, float]:
    """Return timing statistics for `fn`, in milliseconds.

    On CUDA this defers to `triton.testing.do_bench` when Triton is installed,
    since that is the reference implementation the kernel community compares
    against and it handles L2 flushing and CUDA-event timing for us.

    `inner_reps > 1` runs the op that many times between synchronisations and
    divides. This exists because a decode-shaped launch (one row, a few
    kilobytes) finishes far faster than a command buffer can be submitted and
    waited on -- roughly 0.2 ms on Apple silicon. Timed one-at-a-time, every
    such kernel reports the same number and the measurement says nothing about
    the kernel. Amortising the submission is the only way to see the work.

    The tradeoff is real and worth stating: with `inner_reps > 1` the input
    stays resident in cache across replicates, so the result is a warm-cache
    figure. For decode that is arguably the honest case anyway -- the hidden
    state was written by the previous layer microseconds earlier -- but it is
    not comparable to the cold-cache prefill numbers, and the reports keep the
    two regimes in separate tables because of it.
    """
    if backend == "cuda" and inner_reps == 1:
        try:
            from triton.testing import do_bench

            quantiles = [0.5, 0.1, 0.9]
            med, low, high = do_bench(fn, warmup=warmup, rep=iters, quantiles=quantiles)
            return {
                "median_ms": med,
                "p10_ms": low,
                "p90_ms": high,
                "timer": "triton.do_bench",
                "inner_reps": 1,
            }
        except ImportError:
            pass

    # Flushing between replicates is pointless when the replicates deliberately
    # share a warm cache.
    cache = _l2_flush_buffer(backend) if (flush_l2 and inner_reps == 1) else None

    for _ in range(warmup):
        fn()
    _sync(backend)

    samples: list[float] = []
    for _ in range(iters):
        if cache is not None:
            _dirty(cache, backend)
        _sync(backend)
        t0 = time.perf_counter()
        for _ in range(inner_reps):
            fn()
        _sync(backend)
        samples.append((time.perf_counter() - t0) * 1e3 / inner_reps)

    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[int(0.1 * (len(samples) - 1))],
        "p90_ms": samples[int(0.9 * (len(samples) - 1))],
        "timer": "perf_counter+sync",
        "inner_reps": inner_reps,
    }


def _l2_flush_buffer(backend: str):
    """A buffer comfortably larger than any last-level cache we target."""
    n = 64 * 1024 * 1024 // 4  # 64 MiB of fp32
    if backend in ("cuda", "mps"):
        import torch

        return torch.empty(n, dtype=torch.float32, device=backend)
    if backend == "mlx":
        import mlx.core as mx

        return mx.zeros((n,), dtype=mx.float32)
    return None


def _dirty(buf, backend: str) -> None:
    if backend == "mlx":
        import mlx.core as mx

        mx.eval(mx.sum(buf))
    else:
        buf.zero_()


def throughput(bytes_moved: int, ms: float) -> float:
    """Effective bandwidth in GB/s (1 GB = 1e9 bytes), matching vendor convention."""
    return bytes_moved / (ms * 1e-3) / 1e9
