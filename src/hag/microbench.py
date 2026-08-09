"""Measure what the memory system will actually give you.

Datasheet bandwidth is a ceiling nobody reaches. A saturating copy kernel is a
much more honest denominator for "percent of peak", so every report in this
repo carries both: percent of datasheet, and percent of measured copy.
"""

from __future__ import annotations

from . import devices, timing


def measured_copy_bandwidth_gbs(backend: str, mib: int = 256) -> float:
    """Peak achievable bandwidth from a large device-to-device copy.

    A copy moves `2 * n` bytes: one read, one write. Sized well past any
    last-level cache so the number reflects DRAM, not SRAM.
    """
    n = mib * 1024 * 1024 // 4  # fp32 elements

    if backend in ("cuda", "mps"):
        import torch

        src = torch.randn(n, dtype=torch.float32, device=backend)
        dst = torch.empty_like(src)

        def run():
            dst.copy_(src)

    elif backend == "mlx":
        import mlx.core as mx

        src = mx.random.normal((n,), dtype=mx.float32)
        mx.eval(src)

        def run():
            mx.eval(src + 0.0)

    else:
        import torch

        src = torch.randn(n, dtype=torch.float32)
        dst = torch.empty_like(src)

        def run():
            dst.copy_(src)

    stats = timing.bench_ms(run, backend, warmup=5, iters=30, flush_l2=False)
    return timing.throughput(2 * n * 4, stats["median_ms"])


def dispatch_floor_ms(backend: str) -> float:
    """Wall time for the smallest possible kernel launch.

    This is the cost of submitting a command buffer and waiting on it, with
    essentially no work attached. Any measured kernel time near this figure is
    reporting the runtime's launch path, not the kernel -- which is exactly the
    situation at decode, where a single row of hidden state is a few kilobytes
    and the arithmetic is over before the submission has finished.

    Reporting this number is what lets the results tables say "below the
    measurement floor" instead of quietly presenting launch overhead as a
    kernel benchmark.
    """
    if backend in ("cuda", "mps"):
        import torch

        a = torch.zeros(1, device=backend)

        def run():
            a.add_(1.0)

    elif backend == "mlx":
        import mlx.core as mx

        a = mx.zeros((1,))
        mx.eval(a)

        def run():
            mx.eval(a + 1.0)

    else:
        return 0.0

    stats = timing.bench_ms(run, backend, warmup=20, iters=100, flush_l2=False)
    return stats["median_ms"]


def summary(backend: str | None = None) -> dict:
    backend = backend or devices.default_backend()
    info = devices.describe(backend)
    measured = measured_copy_bandwidth_gbs(backend)
    spec = info.get("spec")
    info["measured_copy_gbs"] = round(measured, 1)
    if spec:
        info["copy_pct_of_datasheet"] = round(100 * measured / spec["peak_bandwidth_gbs"], 1)
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(summary(), indent=2))
