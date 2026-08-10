"""Kernel-level profile using PyTorch's own profiler.

`nsys` gives a better timeline, but it is an apt package that is simply absent
from some environments, a stock Colab runtime among them, and a profiling step
that depends on a package that may not install is not a profiling step.
This module answers the same question with nothing but torch:

    which kernels consume the time, and is this workload memory-bound?

Run it first. Reach for `scripts/profile_nsys.sh` when you want the timeline
view of the gaps *between* kernels, which is the one thing this cannot show.

    python -m hag.profile_torch --model Qwen/Qwen2.5-1.5B

Prefill and decode are wrapped in separate `record_function` ranges, so the
summary can be read per regime rather than as one blended average.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import devices, models

#: record_function range names, which wrap kernels rather than being kernels.
REGIME_RANGES = frozenset({"prefill", "decode"})


def _is_device_kernel(name: str) -> bool:
    """True for an actual CUDA kernel rather than a dispatcher-level op.

    `key_averages()` returns both, nested: `aten::mm` is an ATen op whose GPU
    time is spent inside `gemv2T_kernel_val`. Summing the two double-counts.
    Only the device kernels partition the GPU timeline, so shares are computed
    over those alone.
    """
    return not name.startswith(("aten::", "cudaLaunch", "Optimizer", "autograd::"))


def _self_device_us(evt) -> float:
    """Self GPU time for one event, across torch's renames of the field."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(evt, attr, None)
        if value:
            return float(value)
    return 0.0


def _total_device_us(evt) -> float:
    for attr in ("device_time_total", "cuda_time_total"):
        value = getattr(evt, attr, None)
        if value:
            return float(value)
    return 0.0


def profile_model(model, input_ids, backend: str, new_tokens: int, out_dir: Path) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU]
    if backend == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def sync():
        if backend == "cuda":
            torch.cuda.synchronize()
        elif backend == "mps":
            torch.mps.synchronize()

    with torch.inference_mode():
        # Warm up at the shapes that will be profiled, for the same reason the
        # benchmark does: autotuning inside the profiled region would be
        # attributed to the kernels rather than to the tuning.
        warm = model(input_ids, use_cache=True)
        model(
            warm.logits[:, -1:].argmax(dim=-1),
            past_key_values=warm.past_key_values,
            use_cache=True,
        )
        del warm
        sync()

        with profile(activities=activities, record_shapes=True) as prof:
            with record_function("prefill"):
                out = model(input_ids, use_cache=True)
                sync()

            past = out.past_key_values
            tok = out.logits[:, -1:].argmax(dim=-1)

            with record_function("decode"):
                for _ in range(new_tokens):
                    step = model(tok, past_key_values=past, use_cache=True)
                    past = step.past_key_values
                    tok = step.logits[:, -1:].argmax(dim=-1)
                sync()

    has_device_time = any(_self_device_us(e) for e in prof.key_averages())
    sort_key = "self_device_time_total" if has_device_time else "self_cpu_time_total"
    try:
        table = prof.key_averages().table(sort_by=sort_key, row_limit=25)
    except (KeyError, AssertionError):
        # Older torch spells the GPU column differently.
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
    print(table)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace = out_dir / f"torch_trace_{stamp}.json"
    prof.export_chrome_trace(str(trace))

    # `prefill` and `decode` are record_function ranges, not kernels. They span
    # everything inside them, so counting them alongside the kernels they
    # contain double-counts the total and makes every kernel's share look
    # smaller than it is. torch prints the range at 268% of self CUDA time for
    # exactly this reason. They are reported separately.
    ranges: dict[str, float] = {}
    events = []
    for evt in prof.key_averages():
        self_us = _self_device_us(evt) if has_device_time else float(evt.self_cpu_time_total)
        if self_us <= 0:
            continue
        if evt.key in REGIME_RANGES:
            ranges[evt.key] = round(self_us, 1)
            continue
        if has_device_time and not _is_device_kernel(evt.key):
            continue
        events.append(
            {
                "name": evt.key,
                "count": int(evt.count),
                "self_us": round(self_us, 1),
                "total_us": round(
                    _total_device_us(evt) if has_device_time else float(evt.cpu_time_total), 1
                ),
            }
        )
    events.sort(key=lambda e: e["self_us"], reverse=True)
    total = sum(e["self_us"] for e in events) or 1.0
    for e in events:
        e["pct_of_self_time"] = round(100 * e["self_us"] / total, 2)

    # Op dispatches per forward pass. Each one costs CPU time whether or not
    # the GPU has anything to do, which is the whole question at decode.
    forwards = new_tokens + 1  # one prefill plus one per decode step
    dispatches = sum(
        int(e.count) for e in prof.key_averages() if e.key.startswith("aten::")
    )

    return {
        "timed_on": "gpu" if has_device_time else "cpu",
        "trace": str(trace),
        "regime_us": ranges,
        "kernel_self_us_total": round(total, 1),
        "decode_steps": new_tokens,
        "dispatches_per_forward": round(dispatches / forwards),
        "top_kernels": events[:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--out", default="profiles")
    args = ap.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Needs torch:  pip install -e '.[e2e]'") from exc

    backend = devices.default_backend()
    device = "cuda" if backend == "cuda" else ("mps" if backend == "mps" else "cpu")
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    info = devices.describe(backend)
    print(f"device : {info['device_name']}  [{backend}]")
    print(f"model  : {args.model}\n")

    model, tokenizer = models.load_model_and_tokenizer(args.model, dt, device)
    input_ids = torch.randint(
        0, models.vocab_size(tokenizer, model), (1, args.prompt_tokens),
        device=device, dtype=torch.long,
    )

    out_dir = Path(args.out)
    summary = profile_model(model, input_ids, backend, args.new_tokens, out_dir)
    summary["device"] = info
    summary["model"] = args.model

    path = out_dir / "torch_profile_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nwrote {path}")
    print(f"wrote {summary['trace']}  (open at chrome://tracing or perfetto.dev)")
    if summary["timed_on"] == "cpu":
        print(
            "\nNo GPU activity was captured, so the table above is CPU time. On a "
            "CUDA device it reports per-kernel GPU time instead."
        )
    else:
        print(
            "\nRead the top of the table. If elementwise kernels (add, mul, silu, "
            "norms) outrank the GEMMs, the workload is memory-bound and fusion is "
            "the right lever."
        )


if __name__ == "__main__":
    main()
