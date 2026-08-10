"""Is decode limited by bandwidth, by compute, or by the CPU?

This module exists because the answer changed the project's direction. The plan
was to write a GEMV kernel next, since matmul is the largest share of decode GPU
time. Putting the measured numbers against the roofline first said not to:

    GEMV kernel time per decode step   12.1 ms
    bandwidth floor for the weights    12.8 ms

cuBLAS is already at the memory wall, so a hand-written GEMV has nothing to
take. The wall clock was 33.2 ms per token, meaning the GPU sat idle for roughly
two thirds of decode while the CPU issued about 1200 op dispatches per token.

Decode on this machine is dispatch-bound, not bandwidth-bound. That is a
different problem with a different fix, and the only reason to know it is having
done the arithmetic before writing the kernel.

    python -m hag.roofline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Kernel-name fragments that identify matrix-vector work, which is what decode
#: spends its GPU time on. GEMM kernels (`s1688gemm`) are prefill's.
GEMV_MARKERS = ("gemv",)


def _gemv_us(profile: dict) -> float:
    return sum(
        k["self_us"]
        for k in profile.get("top_kernels", [])
        if any(m in k["name"].lower() for m in GEMV_MARKERS)
    )


def analyse(profile: dict, e2e: dict, ops: dict) -> dict | None:
    """Place measured decode against the bandwidth roofline.

    Every input is a committed measurement. Nothing here is a model of what the
    hardware should do; it is arithmetic on what it did.
    """
    if profile.get("timed_on") != "gpu":
        return None

    weight_bytes = e2e.get("weight_bytes")
    decode_steps = profile.get("decode_steps")
    decode_wall_us = (profile.get("regime_us") or {}).get("decode")
    copy_gbs = ops.get("measured_copy_gbs")
    tok_per_s = (e2e.get("baseline") or {}).get("decode_tok_per_s")
    if not all((decode_wall_us, copy_gbs, tok_per_s)):
        return None

    out: dict = {
        "device": profile["device"]["device_name"],
        "model": profile.get("model"),
        "measured_decode_ms_per_token": round(1e3 / tok_per_s, 2),
        "measured_copy_gbs": copy_gbs,
    }

    # How much of decode the GPU was actually busy for. The numerator is every
    # kernel in the profile including prefill's, so this is an upper bound: the
    # true figure is lower, which only strengthens the conclusion.
    kernel_us = profile.get("kernel_self_us_total")
    if kernel_us:
        out["gpu_busy_fraction_upper_bound"] = round(kernel_us / decode_wall_us, 3)

    if weight_bytes and decode_steps:
        floor_ms = weight_bytes / (copy_gbs * 1e9) * 1e3
        gemv_ms = _gemv_us(profile) / decode_steps / 1e3
        out.update(
            {
                "weight_bytes": weight_bytes,
                "bandwidth_floor_ms_per_token": round(floor_ms, 2),
                "bandwidth_ceiling_tok_per_s": round(1e3 / floor_ms, 1),
                "measured_fraction_of_roofline": round(floor_ms / (1e3 / tok_per_s), 3),
                "gemv_ms_per_decode_step": round(gemv_ms, 2),
                # Above 1.0 means the GEMV beats a copy, which it can: a copy is
                # one read per write, a GEMV is almost pure read, and reads
                # sustain better. It says the kernel is read-bandwidth-bound.
                "gemv_fraction_of_roofline": round(floor_ms / gemv_ms, 3) if gemv_ms else None,
            }
        )

    dispatches = profile.get("dispatches_per_forward")
    if dispatches:
        out["op_dispatches_per_token"] = dispatches
        out["implied_cpu_launch_ms_at_20us"] = round(dispatches * 20 / 1000, 1)

    return out


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--profile", default=str(REPO / "profiles" / "torch_profile_summary.json"))
    args = ap.parse_args()

    profile = _load(Path(args.profile))
    results = Path(args.results)
    if not profile:
        raise SystemExit("No profile found. Run `make profile` on a CUDA device first.")

    dev = profile.get("device", {}).get("device_name", "").lower().replace(" ", "-")
    e2e = next((_load(p) for p in results.glob("e2e_*.json") if dev in p.name), {})
    ops = next((_load(p) for p in results.glob("ops_*.json") if dev in p.name), {})

    report = analyse(profile, e2e, ops)
    if report is None:
        raise SystemExit(
            "Not enough data. Needs a GPU profile plus matching ops and e2e runs "
            "from the same device."
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
