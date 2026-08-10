"""Op-level benchmark: fused kernel vs. the framework's unfused baseline.

Emits JSON to `results/`. Nothing in this repo hand-writes a performance
number; `hag.report` renders the tables in the README straight out of these
files, so a claim in the README always has a run behind it.

    python -m hag.bench_ops --dtype fp16
    python -m hag.bench_ops --backend cuda --out results/ops_t4.json

Shapes are chosen to straddle the two regimes that behave completely
differently:

* `rows = 1..8`   decode. One token at a time, latency-bound, and the GPU
                  spends most of its life waiting on memory.
* `rows = 512+`   prefill. Enough parallelism to actually saturate the
                  memory system.

A kernel that wins in one regime and not the other is the normal outcome, not
a failure, and the report keeps them separate for that reason.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import devices, microbench, reference, timing

#: (hidden_size, label) for Llama-3.2-1B, Qwen2.5-1.5B, Llama-3-8B.
HIDDEN_SIZES = [(2048, "1B"), (1536, "1.5B"), (4096, "8B")]
#: SwiGLU intermediate widths for the same three models.
INTERMEDIATE_SIZES = [(8192, "1B"), (8960, "1.5B"), (14336, "8B")]
ROW_COUNTS = [1, 8, 512, 2048]


#: Decode-shaped launches finish faster than a command buffer can be submitted,
#: so they are timed in amortised batches. See `hag.timing.bench_ms`.
DECODE_INNER_REPS = 200


def _inner_reps(m: int) -> int:
    return DECODE_INNER_REPS if m <= 8 else 1


def _dtype(backend: str, name: str):
    if backend == "mlx":
        import mlx.core as mx

        return {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[name]
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _itemsize(backend: str, name: str) -> int:
    return {"fp16": 2, "bf16": 2, "fp32": 4}[name]


# --------------------------------------------------------------------------
# CUDA / Triton
# --------------------------------------------------------------------------


def _run_cuda(dtype_name: str, warmup: int, iters: int) -> list[dict]:
    import torch

    from .kernels.triton import rmsnorm as tri_rmsnorm
    from .kernels.triton import swiglu as tri_swiglu

    dt = _dtype("cuda", dtype_name)
    isz = _itemsize("cuda", dtype_name)
    rows_out: list[dict] = []

    for n, label in HIDDEN_SIZES:
        w = torch.randn(n, dtype=dt, device="cuda")
        for m in ROW_COUNTS:
            x = torch.randn(m, n, dtype=dt, device="cuda")
            res = torch.randn(m, n, dtype=dt, device="cuda")

            reps = _inner_reps(m)
            base = timing.bench_ms(
                lambda x=x, res=res, w=w: reference.rmsnorm_residual(x, res, w),
                "cuda", warmup, iters, inner_reps=reps,
            )
            fused = timing.bench_ms(
                lambda x=x, res=res, w=w: tri_rmsnorm.rmsnorm_residual(x, res, w),
                "cuda", warmup, iters, inner_reps=reps,
            )
            rows_out.append(
                _record(
                    "rmsnorm_residual", label, m, n, dtype_name,
                    base, fused,
                    reference.ideal_bytes_rmsnorm_residual(m, n, isz),
                )
            )

    for n, label in INTERMEDIATE_SIZES:
        for m in ROW_COUNTS:
            g = torch.randn(m, n, dtype=dt, device="cuda")
            u = torch.randn(m, n, dtype=dt, device="cuda")

            reps = _inner_reps(m)
            base = timing.bench_ms(
                lambda g=g, u=u: reference.swiglu(g, u), "cuda", warmup, iters, inner_reps=reps
            )
            fused = timing.bench_ms(
                lambda g=g, u=u: tri_swiglu.swiglu(g, u), "cuda", warmup, iters, inner_reps=reps
            )
            rows_out.append(
                _record(
                    "swiglu", label, m, n, dtype_name,
                    base, fused,
                    reference.ideal_bytes_swiglu(m, n, isz),
                )
            )

    return rows_out


# --------------------------------------------------------------------------
# Metal / MLX
# --------------------------------------------------------------------------


def _run_mlx(dtype_name: str, warmup: int, iters: int) -> list[dict]:
    import mlx.core as mx

    from .kernels.metal import rmsnorm as mtl_rmsnorm
    from .kernels.metal import swiglu as mtl_swiglu

    dt = _dtype("mlx", dtype_name)
    isz = _itemsize("mlx", dtype_name)
    rows_out: list[dict] = []

    def timed(fn):
        # MLX is lazy: without the eval we would be timing graph construction.
        # `fn` binds its tensors as default arguments at definition time, so the
        # closure cannot drift onto the next loop iteration's shapes.
        def run():
            mx.eval(fn())

        return run

    for n, label in HIDDEN_SIZES:
        w = mx.random.normal((n,)).astype(dt)
        for m in ROW_COUNTS:
            x = mx.random.normal((m, n)).astype(dt)
            res = mx.random.normal((m, n)).astype(dt)
            mx.eval(x, res, w)

            reps = _inner_reps(m)
            base = timing.bench_ms(
                timed(lambda x=x, res=res, w=w: mtl_rmsnorm.rmsnorm_residual_reference(x, res, w)),
                "mlx", warmup, iters, inner_reps=reps,
            )
            fused = timing.bench_ms(
                timed(lambda x=x, res=res, w=w: mtl_rmsnorm.rmsnorm_residual(x, res, w)),
                "mlx", warmup, iters, inner_reps=reps,
            )
            rows_out.append(
                _record(
                    "rmsnorm_residual", label, m, n, dtype_name,
                    base, fused,
                    reference.ideal_bytes_rmsnorm_residual(m, n, isz),
                )
            )

    for n, label in INTERMEDIATE_SIZES:
        for m in ROW_COUNTS:
            g = mx.random.normal((m, n)).astype(dt)
            u = mx.random.normal((m, n)).astype(dt)
            mx.eval(g, u)

            reps = _inner_reps(m)
            base = timing.bench_ms(
                timed(lambda g=g, u=u: mtl_swiglu.swiglu_reference(g, u)),
                "mlx", warmup, iters, inner_reps=reps,
            )
            fused = timing.bench_ms(
                timed(lambda g=g, u=u: mtl_swiglu.swiglu(g, u)),
                "mlx", warmup, iters, inner_reps=reps,
            )
            rows_out.append(
                _record(
                    "swiglu", label, m, n, dtype_name,
                    base, fused,
                    reference.ideal_bytes_swiglu(m, n, isz),
                )
            )

    return rows_out


def _record(op, model, m, n, dtype_name, base, fused, ideal_bytes) -> dict:
    return {
        "op": op,
        "model_class": model,
        "rows": m,
        "width": n,
        "dtype": dtype_name,
        "regime": "decode" if m <= 8 else "prefill",
        "baseline_ms": round(base["median_ms"], 6),
        "fused_ms": round(fused["median_ms"], 6),
        "speedup": round(base["median_ms"] / fused["median_ms"], 3),
        "ideal_bytes": ideal_bytes,
        "fused_gbs": round(timing.throughput(ideal_bytes, fused["median_ms"]), 1),
        "baseline_gbs": round(timing.throughput(ideal_bytes, base["median_ms"]), 1),
        "timer": fused["timer"],
        "inner_reps": fused.get("inner_reps", 1),
    }


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="auto", choices=["auto", "cuda", "mlx"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None, help="JSON path (default: results/ops_<device>.json)")
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        avail = devices.available_backends()
        backend = "cuda" if "cuda" in avail else ("mlx" if "mlx" in avail else None)
        if backend is None:
            raise SystemExit(
                "No GPU backend found. This benchmark needs CUDA (for Triton) or "
                "Apple Metal (for MLX)."
            )

    info = devices.describe(backend)
    peak_copy = microbench.measured_copy_bandwidth_gbs(backend)
    floor_ms = microbench.dispatch_floor_ms(backend)
    print(f"device : {info['device_name']}  [{backend}]")
    print(f"copy   : {peak_copy:.1f} GB/s measured")
    print(f"floor  : {floor_ms * 1e3:.0f} us per dispatch")

    runner = _run_cuda if backend == "cuda" else _run_mlx
    rows = runner(args.dtype, args.warmup, args.iters)

    for r in rows:
        r["fused_pct_of_measured_peak"] = round(100 * r["fused_gbs"] / peak_copy, 1)
        # Within 2x of the launch floor, the measurement is describing the
        # runtime rather than the kernel. Flagged rather than dropped, because
        # "this op is too small to measure this way" is itself a finding.
        r["dispatch_bound"] = r["fused_ms"] < 2 * floor_ms
        spec = info.get("spec")
        if spec:
            r["fused_pct_of_datasheet"] = round(
                100 * r["fused_gbs"] / spec["peak_bandwidth_gbs"], 1
            )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "python": platform.python_version(),
        "device": info,
        "measured_copy_gbs": round(peak_copy, 1),
        "dispatch_floor_ms": round(floor_ms, 6),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": rows,
    }

    slug = info["device_name"].lower().replace(" ", "-").replace("(r)", "").strip("-")
    out = Path(args.out or f"results/ops_{slug}_{args.dtype}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    header = (
        f"\n{'op':<18}{'shape':>14}{'base ms':>10}"
        f"{'fused ms':>10}{'speedup':>9}{'GB/s':>8}{'%peak':>7}"
    )
    print(header)
    for r in rows:
        shape = f"{r['rows']}x{r['width']}"
        print(
            f"{r['op']:<18}{shape:>14}"
            f"{r['baseline_ms']:>10.4f}{r['fused_ms']:>10.4f}"
            f"{r['speedup']:>8.2f}x{r['fused_gbs']:>8.0f}"
            f"{r['fused_pct_of_measured_peak']:>6.0f}%"
            f"{'  <- dispatch-bound' if r['dispatch_bound'] else ''}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
