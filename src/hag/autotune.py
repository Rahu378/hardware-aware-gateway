"""Measure the fusion crossover on *this* machine instead of trusting a constant.

    python -m hag.autotune

`SWIGLU_MIN_ROWS` was set from a T4 sweep: fusing loses at 8 rows and wins at
32. That number is not a property of the kernel. It is the row count at which
the traffic saved exceeds Triton's launch cost, and launch cost varies by an
order of magnitude across devices. The measured launch floor is 6 to 10 us on a
T4 and 177 us on an M3, so a threshold calibrated on one is close to
meaningless on the other.

This runs a short sweep at import-time cost of nothing, writes the answer to a
cache keyed by device name, and lets the kernel read it. Roughly two seconds,
against a benchmark sweep that takes minutes, because it only needs the sign of
the comparison and not a publishable number.

Calibration is explicit rather than automatic. A library that silently
benchmarks the GPU the first time it is imported is a library that produces
mysterious pauses and non-reproducible thresholds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Bracketing sweep. Powers of two around the crossovers seen so far, which is
#: enough to locate a sign change without pretending to more precision.
CANDIDATE_ROWS = (1, 8, 16, 32, 64, 128, 256, 512)

#: Widest of the intermediate sizes in the sweep. Fusion's advantage grows with
#: width, so calibrating at the narrowest would set a threshold that is too
#: conservative for every other shape.
CALIBRATION_WIDTH = 8960


def cache_path() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(root) / "hardware-aware-gateway" / "calibration.json"


def _load_cache() -> dict:
    path = cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # A corrupt cache should cost a recalibration, not a crash.
        return {}


def _save_cache(cache: dict) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2) + "\n")


def measure_swiglu_crossover(iters: int = 30, warmup: int = 10) -> int:
    """Smallest row count at which the fused kernel beats eager, and keeps beating it.

    "Keeps beating" matters: a single crossing can be noise, and a threshold set
    on a lucky sample would send decode-shaped work through the slower path.
    """
    import torch

    from . import reference, timing
    from .kernels.triton import swiglu as tri

    wins: list[int] = []
    losses: list[int] = []
    for rows in CANDIDATE_ROWS:
        g = torch.randn(rows, CALIBRATION_WIDTH, dtype=torch.float16, device="cuda")
        u = torch.randn_like(g)
        reps = 200 if rows <= 8 else 1
        base = timing.bench_ms(
            lambda g=g, u=u: reference.swiglu(g, u), "cuda", warmup, iters, inner_reps=reps
        )
        fused = timing.bench_ms(
            lambda g=g, u=u: tri.swiglu_triton(g, u), "cuda", warmup, iters, inner_reps=reps
        )
        (wins if fused["median_ms"] < base["median_ms"] else losses).append(rows)

    return next((w for w in sorted(wins) if not any(x > w for x in losses)), max(CANDIDATE_ROWS))


def calibrate(force: bool = False, apply: bool = True) -> dict:
    """Measure this device's crossover, cache it, and optionally apply it."""
    from . import devices
    from .kernels.triton import swiglu as tri

    name = devices.device_name("cuda")
    cache = _load_cache()

    if not force and name in cache:
        entry = cache[name]
        entry["source"] = "cache"
    else:
        entry = {"swiglu_min_rows": measure_swiglu_crossover(), "source": "measured"}
        cache[name] = {k: v for k, v in entry.items() if k != "source"}
        _save_cache(cache)

    entry["device"] = name
    entry["default"] = tri.SWIGLU_MIN_ROWS
    if apply:
        tri.SWIGLU_MIN_ROWS = entry["swiglu_min_rows"]
    return entry


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "Calibration needs a CUDA device: it is measuring this machine's "
            "launch cost against its bandwidth, and neither is portable."
        )

    from .kernels.triton import swiglu as tri

    before = tri.SWIGLU_MIN_ROWS
    entry = calibrate(force=True)
    print(f"device            : {entry['device']}")
    print(f"compiled-in default: {before} rows")
    print(f"measured crossover : {entry['swiglu_min_rows']} rows")
    print(f"cached at          : {cache_path()}")
    if entry["swiglu_min_rows"] != before:
        print(
            f"\nThe default is wrong for this device by "
            f"{entry['swiglu_min_rows'] / before:.1f}x. Call hag.calibrate() at "
            "startup to use the measured value."
        )
    else:
        print("\nThe compiled-in default matches this device.")


if __name__ == "__main__":
    main()
