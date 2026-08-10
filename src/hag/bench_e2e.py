"""End-to-end benchmark: tokens/sec and memory footprint on a real model.

Op-level speedups are the interesting engineering; this is the number that
decides whether any of it mattered. A 2x win on a kernel that occupies 4% of
the forward pass moves the headline by 2%, and saying so plainly is worth more
than a chart that hides it.

    python -m hag.bench_e2e --model Qwen/Qwen2.5-0.5B          # fits an 8 GB Mac
    python -m hag.bench_e2e --model meta-llama/Meta-Llama-3-8B # needs a real GPU

Prefill and decode are timed separately, because they are different machines:
prefill is compute-bound and batched, decode is memory-bound and serial. A
kernel change usually moves exactly one of them.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import devices, models

#: Two-sided t critical values at 95%, indexed by degrees of freedom.
#: Small table rather than a scipy dependency; this benchmark should run on a
#: bare Colab runtime with nothing but torch installed.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
    24: 2.064, 30: 2.042, 40: 2.021, 60: 2.000,
}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("inf")
    for k in sorted(_T_CRIT_95):
        if df <= k:
            return _T_CRIT_95[k]
    return 1.96


def _paired_stats(base: list[dict], fused: list[dict], key: str) -> dict:
    """Paired analysis of baseline vs fused, which is how they were measured.

    Baseline and fused run adjacently inside each repeat, so they share whatever
    the machine was doing at that moment. Differencing within a pair cancels
    that drift, and the test is far more sensitive than comparing the two
    distributions independently. Comparing ranges throws the pairing away, which
    on a shared GPU means throwing away most of the signal.

    `resolved` is a two-sided 95% t-test on the paired differences: is the mean
    difference distinguishable from zero given how much the differences scatter.
    """
    diffs = [f[key] - b[key] for b, f in zip(base, fused, strict=True)]
    n = len(diffs)
    if n < 2:
        return {"samples": n, "resolved": False, "reason": "need at least 2 repeats"}

    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = var ** 0.5
    sem = sd / (n ** 0.5) if sd else 0.0
    t = mean / sem if sem else float("inf") if mean else 0.0
    crit = _t_crit(n - 1)

    # Repeats needed for this effect size to clear the bar, as a hint rather
    # than a promise: if the effect is real and this small, it takes this many.
    needed = None
    if sd and mean:
        needed = int((2.1 * sd / abs(mean)) ** 2) + 1

    return {
        "samples": n,
        "mean_diff": round(mean, 4),
        "sd_diff": round(sd, 4),
        "t": round(t, 3),
        "t_crit_95": crit,
        "resolved": abs(t) > crit,
        "repeats_needed_estimate": needed,
        "diffs": [round(d, 3) for d in diffs],
    }


def _summarise(samples: list[dict], key: str) -> dict:
    """Median plus observed range for one metric across repeats."""
    vals = sorted(s[key] for s in samples)
    n = len(vals)
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {
        "median": round(median, 3),
        "min": round(vals[0], 3),
        "max": round(vals[-1], 3),
        "samples": n,
    }


def _peak_memory_bytes(backend: str) -> int:
    if backend == "cuda":
        import torch

        return torch.cuda.max_memory_allocated()
    if backend == "mps":
        import torch

        return torch.mps.current_allocated_memory()
    import resource

    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys

    return rss if sys.platform == "darwin" else rss * 1024


def _reset_peak_memory(backend: str) -> None:
    if backend == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()


def patch_model(model, backend: str) -> list[str]:
    """Swap the fused kernels into a HuggingFace Llama-family model in place.

    Returns the list of modules actually patched, so the report can state what
    was and was not replaced rather than implying whole-model coverage.
    """
    if backend != "cuda":
        return []

    import torch

    from .kernels.triton import rmsnorm as tri_rmsnorm
    from .kernels.triton import swiglu as tri_swiglu

    patched: list[str] = []

    for module in model.modules():
        cls = type(module).__name__

        if cls.endswith("RMSNorm") and hasattr(module, "weight"):
            eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))

            def rms_forward(self, hidden_states, _eps=eps):
                return tri_rmsnorm.rmsnorm(hidden_states, self.weight, _eps)

            module.forward = rms_forward.__get__(module, type(module))
            patched.append(f"{cls}.forward -> triton.rmsnorm")

        elif cls.endswith("MLP") and all(
            hasattr(module, p) for p in ("gate_proj", "up_proj", "down_proj")
        ):

            def mlp_forward(self, x):
                # The dispatching entry point, not the raw kernel: below
                # SWIGLU_MIN_ROWS this falls back to eager, which is what keeps
                # the prefill win without paying the decode penalty.
                return self.down_proj(tri_swiglu.swiglu(self.gate_proj(x), self.up_proj(x)))

            module.forward = mlp_forward.__get__(module, type(module))
            patched.append(
                f"{cls}.forward -> triton.swiglu (fused at >= "
                f"{tri_swiglu.SWIGLU_MIN_ROWS} rows)"
            )

    del torch
    return patched


def run_once(model, tokenizer, backend: str, prompt_tokens: int, new_tokens: int) -> dict:
    import torch

    _reset_peak_memory(backend)
    device = "cuda" if backend == "cuda" else ("mps" if backend == "mps" else "cpu")

    input_ids = torch.randint(
        0, models.vocab_size(tokenizer, model), (1, prompt_tokens),
        device=device, dtype=torch.long,
    )

    def sync():
        if backend == "cuda":
            torch.cuda.synchronize()
        elif backend == "mps":
            torch.mps.synchronize()

    with torch.inference_mode():
        # Warm up at the *exact* shapes that will be timed. A short warmup is
        # worse than none here: `_swiglu_fwd` is autotuned and keyed on element
        # count, so a warmup at a different sequence length leaves the real
        # prefill to pay for the autotuning sweep inside the timed region.
        # Decode is warmed separately for the same reason: its element count
        # differs from prefill's, and so is a separate autotune key.
        warm = model(input_ids, use_cache=True)
        model(
            warm.logits[:, -1:].argmax(dim=-1),
            past_key_values=warm.past_key_values,
            use_cache=True,
        )
        del warm
        sync()

        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True)
        sync()
        prefill_s = time.perf_counter() - t0

        past = out.past_key_values
        next_tok = out.logits[:, -1:].argmax(dim=-1)

        t0 = time.perf_counter()
        for _ in range(new_tokens):
            step = model(next_tok, past_key_values=past, use_cache=True)
            past = step.past_key_values
            next_tok = step.logits[:, -1:].argmax(dim=-1)
        sync()
        decode_s = time.perf_counter() - t0

    return {
        "prefill_tokens": prompt_tokens,
        "prefill_s": round(prefill_s, 5),
        "prefill_tok_per_s": round(prompt_tokens / prefill_s, 1),
        "decode_tokens": new_tokens,
        "decode_s": round(decode_s, 5),
        "decode_tok_per_s": round(new_tokens / decode_s, 2),
        "ms_per_output_token": round(decode_s / new_tokens * 1e3, 3),
        "peak_memory_gb": round(_peak_memory_bytes(backend) / 1e9, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument(
        "--repeats", type=int, default=5,
        help="alternating baseline/fused measurements; medians are reported",
    )
    ap.add_argument("--out", default=None)
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
    print(f"model  : {args.model} ({args.dtype})")

    model, tokenizer = models.load_model_and_tokenizer(args.model, dt, device)

    # Decode streams essentially every weight once per token, so this is the
    # numerator of the bandwidth roofline in `hag.roofline`.
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"weights: {weight_bytes / 1e9:.2f} GB")

    unpatched = {m: m.forward for m in model.modules()}

    patched_modules = patch_model(model, backend)
    patched_forward = {m: m.forward for m in model.modules()}

    def restore(mapping):
        for module, fn in mapping.items():
            module.forward = fn

    # Baseline and fused are measured alternately rather than one after the
    # other. A shared GPU drifts: across two runs of this benchmark the
    # *unpatched* model measured 32.24 and then 25.98 tok/s, a 20% swing with
    # nothing changed. Timed in sequence, that drift lands entirely on whichever
    # configuration ran second and is indistinguishable from a result.
    base_samples, fused_samples = [], []
    for i in range(args.repeats):
        restore(unpatched)
        gc.collect()
        base_samples.append(
            run_once(model, tokenizer, backend, args.prompt_tokens, args.new_tokens)
        )
        if patched_modules:
            restore(patched_forward)
            gc.collect()
            fused_samples.append(
                run_once(model, tokenizer, backend, args.prompt_tokens, args.new_tokens)
            )
        print(
            f"  repeat {i + 1}/{args.repeats}: "
            f"baseline {base_samples[-1]['decode_tok_per_s']:.2f}"
            + (f", fused {fused_samples[-1]['decode_tok_per_s']:.2f}" if patched_modules else "")
            + " tok/s decode"
        )
    restore(unpatched)

    baseline = dict(base_samples[-1])
    baseline["decode_tok_per_s_stats"] = _summarise(base_samples, "decode_tok_per_s")
    baseline["prefill_tok_per_s_stats"] = _summarise(base_samples, "prefill_tok_per_s")
    baseline["decode_tok_per_s"] = baseline["decode_tok_per_s_stats"]["median"]
    baseline["prefill_tok_per_s"] = baseline["prefill_tok_per_s_stats"]["median"]

    fused = speedup = None
    if patched_modules:
        fused = dict(fused_samples[-1])
        fused["decode_tok_per_s_stats"] = _summarise(fused_samples, "decode_tok_per_s")
        fused["prefill_tok_per_s_stats"] = _summarise(fused_samples, "prefill_tok_per_s")
        fused["decode_tok_per_s"] = fused["decode_tok_per_s_stats"]["median"]
        fused["prefill_tok_per_s"] = fused["prefill_tok_per_s_stats"]["median"]
        speedup = fused["decode_tok_per_s"] / baseline["decode_tok_per_s"]

        bs, fs = baseline["decode_tok_per_s_stats"], fused["decode_tok_per_s_stats"]
        paired = _paired_stats(base_samples, fused_samples, "decode_tok_per_s")
        baseline["paired_decode"] = paired

        print(f"\nbaseline decode: {bs['median']:.2f} tok/s "
              f"(range {bs['min']:.2f} to {bs['max']:.2f} over {bs['samples']})")
        print(f"fused    decode: {fs['median']:.2f} tok/s "
              f"(range {fs['min']:.2f} to {fs['max']:.2f} over {fs['samples']})")
        print(f"decode speedup : {speedup:.3f}x")
        print(
            f"\npaired difference: {paired['mean_diff']:+.2f} tok/s, "
            f"sd {paired['sd_diff']:.2f}, t = {paired['t']:.2f} "
            f"against t_crit {paired['t_crit_95']:.3f} at 95%"
        )
        if paired["resolved"]:
            print("The difference is resolved: it is larger than the run-to-run scatter.")
        else:
            need = paired.get("repeats_needed_estimate")
            print(
                "NOT RESOLVED. The paired differences scatter more than they differ "
                "from zero,\nso this speedup is not distinguishable from noise on this "
                "machine."
                + (f"\nAn effect this size would need roughly --repeats {need} to clear "
                   "the bar." if need else "")
            )
    else:
        print(
            "\nNo fused kernels applied: the Triton kernels are CUDA-only, so on this\n"
            "device the run above is the unmodified baseline. It is still recorded,\n"
            "as the edge-hardware reference point the cross-platform comparison needs."
        )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "dtype": args.dtype,
        "device": info,
        "patched_modules": sorted(set(patched_modules)),
        "repeats": args.repeats,
        "weight_bytes": weight_bytes,
        "baseline_samples": [s["decode_tok_per_s"] for s in base_samples],
        "fused_samples": [s["decode_tok_per_s"] for s in fused_samples],
        "baseline": baseline,
        "fused": fused,
        "decode_speedup": None if speedup is None else round(speedup, 4),
    }

    slug = info["device_name"].lower().replace(" ", "-")
    model_slug = args.model.split("/")[-1].lower()
    out = Path(args.out or f"results/e2e_{model_slug}_{slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
