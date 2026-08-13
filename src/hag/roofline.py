"""Is decode limited by bandwidth, by compute, or by the CPU?

This module exists because the answer changed the project's direction. The plan
was to write a GEMV kernel next, since matmul is the largest share of decode GPU
time. Putting the measured numbers against the roofline first said not to:

    bandwidth floor for the weights    12.8 ms
    GEMV kernel time per decode step   14.7 ms   (87% of the floor)
    measured wall clock                32.5 ms   (39% of the floor)

A flawless GEMV would recover the 1.9 ms between those first two lines, under
6% of a token. The 17.8 ms between kernel time and wall clock is the CPU
issuing roughly six thousand op dispatches per token at about 3 us each, and
that is 55% of the token.

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
                # Can legitimately exceed 1.0: the floor uses a copy, which is
                # one read per write, while a GEMV is almost pure read and reads
                # sustain better. So a value near or above 1.0 both mean the same
                # thing, that the kernel is at the memory wall.
                "gemv_fraction_of_roofline": round(floor_ms / gemv_ms, 3) if gemv_ms else None,
            }
        )

    dispatches = profile.get("dispatches_per_forward")
    if dispatches and "gemv_ms_per_decode_step" in out:
        # Per-dispatch cost is derived, not assumed. An earlier version of this
        # assumed 20 us and produced 119 ms of implied launch cost for a 32 ms
        # token, which is impossible on its face; the measured figure is about
        # 3 us, the usual cost of a PyTorch eager dispatch.
        gap_ms = out["measured_decode_ms_per_token"] - out["gemv_ms_per_decode_step"]
        out["op_dispatches_per_token"] = dispatches
        out["cpu_gap_ms_per_token"] = round(gap_ms, 1)
        out["cpu_gap_fraction_of_token"] = round(
            gap_ms / out["measured_decode_ms_per_token"], 3
        )
        out["implied_us_per_dispatch"] = round(gap_ms * 1000 / dispatches, 2)

    if "gemv_ms_per_decode_step" in out:
        # What a flawless GEMV would actually be worth, which is the number that
        # decides whether writing one is a good use of a week.
        recoverable = out["gemv_ms_per_decode_step"] - out["bandwidth_floor_ms_per_token"]
        out["perfect_gemv_would_recover_ms"] = round(max(recoverable, 0.0), 2)
        out["perfect_gemv_would_recover_fraction"] = round(
            max(recoverable, 0.0) / out["measured_decode_ms_per_token"], 3
        )

    return out


#: Request shapes to score prefill capture against. Chosen to span the range
#: where the answer might plausibly differ: chat, RAG, and summarisation.
WORKLOADS = ((128, 128), (512, 128), (2048, 128), (4096, 50), (8192, 50))


def prefill_capture_value(profile: dict, e2e: dict, us_per_dispatch: float) -> dict | None:
    """What capturing prefill into a CUDA graph would be worth.

    This cancelled the work, the same way the roofline cancelled the GEMV, and
    for a sharper reason. Dispatch overhead per forward pass is *fixed*: the
    same op count runs whatever the sequence length, because the ops are the
    same and only the tensors get bigger. Prefill compute, meanwhile, scales
    with length.

    So the two ends squeeze it. At a short prompt the overhead is a large share
    of prefill and the speedup looks good, but prefill is a few percent of the
    request. At a long prompt prefill dominates the request, but the fixed
    overhead has become a rounding error against it. There is no prompt length
    where both are true, and the request-level speedup lands near 1.006x
    everywhere.

    Decode had the opposite shape, which is why capturing it was worth 1.72x:
    the same fixed overhead against a step that does one token of work.
    """
    dispatches = profile.get("dispatches_per_forward")
    base = e2e.get("baseline") or {}
    prefill_ms = (base.get("prefill_s") or 0) * 1000
    prefill_tokens = base.get("prefill_tokens")
    decode_ms = 1e3 / base["decode_tok_per_s"] if base.get("decode_tok_per_s") else None
    if not all((dispatches, prefill_ms, prefill_tokens, decode_ms)):
        return None

    overhead_ms = dispatches * us_per_dispatch / 1000
    work_per_token = (prefill_ms - overhead_ms) / prefill_tokens
    rows = []
    for n_in, n_out in WORKLOADS:
        pre = work_per_token * n_in + overhead_ms
        total = pre + n_out * decode_ms
        rows.append(
            {
                "prompt_tokens": n_in,
                "output_tokens": n_out,
                "prefill_ms": round(pre, 1),
                "overhead_share_of_prefill": round(overhead_ms / pre, 3),
                "prefill_speedup": round(pre / (pre - overhead_ms), 3),
                "request_speedup": round(total / (total - overhead_ms), 4),
            }
        )
    return {
        "fixed_dispatch_overhead_ms": round(overhead_ms, 1),
        "dispatches_per_forward": dispatches,
        "best_request_speedup": max(r["request_speedup"] for r in rows),
        "workloads": rows,
    }


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
    if report and report.get("implied_us_per_dispatch"):
        pc = prefill_capture_value(profile, e2e, report["implied_us_per_dispatch"])
        if pc:
            report["prefill_capture_value"] = pc
    if report is None:
        raise SystemExit(
            "Not enough data. Needs a GPU profile plus matching ops and e2e runs "
            "from the same device."
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
