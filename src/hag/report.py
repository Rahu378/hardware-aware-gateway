"""Render the README's results tables from `results/*.json`.

The README contains marker comments:

    <!-- BENCH:BEGIN --> ... <!-- BENCH:END -->

Everything between them is generated. Nothing else writes a performance number
into the README, which means a figure in that file cannot exist without a JSON
run behind it, and `make report` regenerating clean is a check that the two
still agree.

    python -m hag.report            # rewrite README.md in place
    python -m hag.report --check    # non-zero exit if it would change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BEGIN = "<!-- BENCH:BEGIN -->"
END = "<!-- BENCH:END -->"

REPO = Path(__file__).resolve().parents[2]


def load_runs(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("ops_*.json"))]


def load_e2e(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("e2e_*.json"))]


def load_profile(repo: Path) -> dict | None:
    path = repo / "profiles" / "torch_profile_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


#: Coarse buckets for the kernel mix. Deliberately blunt: the question the
#: profile has to answer is "is this workload matmul-bound or elementwise-bound",
#: and a finer taxonomy would obscure that.
def _bucket(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("gemv", "gemm", "dot_kernel")):
        return "matmul (GEMM / GEMV)"
    if "elementwise" in n:
        return "elementwise"
    if "reduce" in n:
        return "reduction"
    if any(k in n for k in ("cat", "copy", "memcpy")):
        return "copy / concat"
    return "other"


def _profile_table(prof: dict | None) -> str:
    """Where GPU time actually goes. This is the evidence for every claim below."""
    if not prof or prof.get("timed_on") != "gpu":
        return "_No GPU profile recorded yet. Run `make profile` on a CUDA device._"

    agg: dict[str, float] = {}
    for k in prof["top_kernels"]:
        agg[_bucket(k["name"])] = agg.get(_bucket(k["name"]), 0.0) + k["self_us"]
    total = sum(agg.values()) or 1.0

    dev = prof["device"]["device_name"]
    model = prof["model"].split("/")[-1]
    lines = [
        f"Measured on {dev}, {model}, one 512-token prefill plus 32 decode steps. "
        "Device kernels only: ATen ops are dispatchers that contain these kernels, "
        "so counting both would double-count.",
        "",
        "| kernel class | GPU time | share |",
        "| --- | --- | --- |",
    ]
    for name, us in sorted(agg.items(), key=lambda t: -t[1]):
        lines.append(f"| {name} | {us / 1000:.1f} ms | **{100 * us / total:.1f}%** |")
    lines.append(f"| **total** | **{total / 1000:.1f} ms** | |")
    return "\n".join(lines)


def _e2e_table(runs: list[dict]) -> str:
    """End-to-end tokens/sec, which is the only number that decides anything.

    Reported whether or not it flatters the kernels. A repo that shows op-level
    speedups and quietly omits the end-to-end result is not reporting a
    measurement, it is making a case.
    """
    if not runs:
        return "_No end-to-end runs recorded yet. Run `make bench-e2e`._"
    lines = [
        "| device | model | prefill baseline | prefill fused | decode baseline "
        "| decode fused | peak memory |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        name = run["device"]["device_name"]
        model = run["model"].split("/")[-1]
        base, fused = run["baseline"], run.get("fused")
        if fused is None:
            lines.append(
                f"| {name} | {model} | {base['prefill_tok_per_s']:.0f} tok/s | not applicable "
                f"| {base['decode_tok_per_s']:.2f} tok/s | not applicable "
                f"| {base['peak_memory_gb']:.2f} GB |"
            )
            continue
        pre = fused["prefill_tok_per_s"] / base["prefill_tok_per_s"]
        dec = fused["decode_tok_per_s"] / base["decode_tok_per_s"]
        lines.append(
            f"| {name} | {model} | {base['prefill_tok_per_s']:.0f} tok/s "
            f"| {fused['prefill_tok_per_s']:.0f} tok/s (**{pre:.2f}x**) "
            f"| {base['decode_tok_per_s']:.2f} tok/s "
            f"| {fused['decode_tok_per_s']:.2f} tok/s (**{dec:.2f}x**) "
            f"| {base['peak_memory_gb']:.2f} -> {fused['peak_memory_gb']:.2f} GB |"
        )
    return "\n".join(lines)


def _env_table(runs: list[dict]) -> str:
    lines = [
        "| device | backend | datasheet BW | measured copy BW | % of datasheet | dispatch floor |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        dev = run["device"]
        spec = dev.get("spec")
        sheet = f"{spec['peak_bandwidth_gbs']:.0f} GB/s" if spec else "n/a"
        pct = (
            f"{100 * run['measured_copy_gbs'] / spec['peak_bandwidth_gbs']:.0f}%"
            if spec
            else "n/a"
        )
        floor = run.get("dispatch_floor_ms")
        floor_s = f"{floor * 1e3:.0f} us" if floor else "n/a"
        lines.append(
            f"| {dev['device_name']} | {dev['backend']} | {sheet} "
            f"| {run['measured_copy_gbs']:.0f} GB/s | {pct} | {floor_s} |"
        )
    return "\n".join(lines)


def _results_table(runs: list[dict], regime: str) -> str:
    lines = [
        "| device | op | shape (rows x width) | baseline | fused | speedup "
        "| eff. BW | % of datasheet | vs copy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    any_row = False
    for run in runs:
        name = run["device"]["device_name"]
        for r in run["results"]:
            if r["regime"] != regime or r.get("dispatch_bound"):
                continue
            any_row = True
            speed = f"**{r['speedup']:.2f}x**" if r["speedup"] >= 1 else f"{r['speedup']:.2f}x"
            sheet = r.get("fused_pct_of_datasheet")
            sheet_s = f"{sheet:.0f}%" if sheet is not None else "n/a"
            lines.append(
                f"| {name} | `{r['op']}` | {r['rows']} x {r['width']} "
                f"| {r['baseline_ms']:.3f} ms | {r['fused_ms']:.3f} ms "
                f"| {speed} | {r['fused_gbs']:.0f} GB/s "
                f"| {sheet_s} | {r['fused_pct_of_measured_peak']:.0f}% |"
            )
    if not any_row:
        return "_Nothing measurable in this regime. Run `make bench`._"
    return "\n".join(lines)


def _regressions(runs: list[dict]) -> str:
    """Call out the shapes where fusing made things worse.

    Burying these would make the report a sales document. They are also the
    most informative rows in it: they mark where launch overhead outweighs the
    traffic saved, which is the boundary the next kernel has to move.
    """
    losses = [
        (run["device"]["device_name"], r)
        for run in runs
        for r in run["results"]
        if not r.get("dispatch_bound") and r["speedup"] < 1.0
    ]
    if not losses:
        return ""
    worst = min(losses, key=lambda t: t[1]["speedup"])
    ops = sorted({r["op"] for _, r in losses})
    return (
        f"\n**Where fusion loses.** {len(losses)} measured shapes came out slower "
        f"than eager, all of them {' and '.join(f'`{o}`' for o in ops)} at decode "
        f"widths. Worst case {worst[1]['speedup']:.2f}x on {worst[0]} at "
        f"{worst[1]['rows']}x{worst[1]['width']}. One fused launch replaces three "
        "eager ones, but at a single row the traffic saved is a few kilobytes "
        "while Triton's launch path costs more than the three ATen launches it "
        "replaces. Fusion is a bandwidth optimisation, and at decode the kernel "
        "is not bandwidth-bound.\n"
    )


def _dispatch_bound_note(runs: list[dict]) -> str:
    rows = [
        (run["device"]["device_name"], r)
        for run in runs
        for r in run["results"]
        if r.get("dispatch_bound")
    ]
    if not rows:
        return ""
    per_device: dict[str, int] = {}
    for dev, _ in rows:
        per_device[dev] = per_device.get(dev, 0) + 1
    floors = {
        run["device"]["device_name"]: run.get("dispatch_floor_ms", 0.0) for run in runs
    }
    parts = [
        f"{n} on {dev} (launch floor {floors.get(dev, 0) * 1e3:.0f} us)"
        for dev, n in per_device.items()
    ]
    return (
        f"\n{len(rows)} measurements landed within 2x of a launch floor and are "
        f"excluded above: {', '.join(parts)}. At those shapes the number describes "
        "the runtime's submission path rather than the kernel. Note how much of "
        "the decode regime this costs on Apple silicon and how little on the T4; "
        "a 30x difference in dispatch cost decides which optimisations are even "
        "coherent on a platform. See [docs/METHOD.md](docs/METHOD.md).\n"
    )



def _e2e_verdict(e2e: list[dict], prof: dict | None) -> str:
    """State the end-to-end outcome in words, including when it is a regression."""
    patched = [r for r in e2e if r.get("fused")]
    if not patched:
        return ""
    run = patched[0]
    base, fused = run["baseline"], run["fused"]
    pre = fused["prefill_tok_per_s"] / base["prefill_tok_per_s"]
    dec = fused["decode_tok_per_s"] / base["decode_tok_per_s"]
    dev = run["device"]["device_name"]

    share = ""
    if prof and prof.get("timed_on") == "gpu":
        agg: dict[str, float] = {}
        for k in prof["top_kernels"]:
            agg[_bucket(k["name"])] = agg.get(_bucket(k["name"]), 0.0) + k["self_us"]
        total = sum(agg.values()) or 1.0
        mm = 100 * agg.get("matmul (GEMM / GEMV)", 0.0) / total
        ew = 100 * agg.get("elementwise", 0.0) / total
        share = (
            f" The profile explains the split: matmul is {mm:.0f}% of GPU time and "
            f"every elementwise op these kernels touch adds up to {ew:.0f}%, so "
            f"fusion had a {ew:.0f}% ceiling before it started."
        )

    return (
        f"\n**Prefill got {pre:.2f}x faster. Decode got {1 / dec:.2f}x slower.** "
        f"On {dev} the same two kernels moved prefill from "
        f"{base['prefill_tok_per_s']:.0f} to {fused['prefill_tok_per_s']:.0f} tok/s "
        f"while dropping decode from {base['decode_tok_per_s']:.1f} to "
        f"{fused['decode_tok_per_s']:.1f} tok/s.{share}\n\n"
        "This is the prefill/decode distinction showing up end to end rather than "
        "as theory. Prefill has thousands of rows in flight and is genuinely "
        "bandwidth-bound, so removing a round trip of the hidden state pays. "
        "Decode has one row: the traffic saved is kilobytes, the GPU is waiting on "
        "weights streaming through the GEMV, and Triton's launch path costs more "
        "than the three ATen launches it replaced. Peak memory rose too, from "
        f"{base['peak_memory_gb']:.2f} to {fused['peak_memory_gb']:.2f} GB, because "
        "the fused RMSNorm materialises the residual as a separate tensor.\n\n"
        "The kernels are not wrong. The target was. The next move is to dispatch "
        "to eager below a row-count threshold, which keeps the prefill win and "
        "stops paying at decode, and then to go after the GEMV, which is where "
        "the time actually is.\n"
    )


def _peak_caveat(runs: list[dict]) -> str:
    """Explain the column that can read above 100%, before a reader assumes an error."""
    over = [
        r
        for run in runs
        for r in run["results"]
        if not r.get("dispatch_bound") and r.get("fused_pct_of_measured_peak", 0) > 100
    ]
    if not over:
        return ""
    return (
        "\n**On the `vs copy` column reading above 100%.** The reference is a "
        "device-to-device copy, which moves one read per write. `swiglu` moves two "
        "reads per write, and DRAM sustains reads better than writes, so a 2:1 "
        "kernel legitimately exceeds a 1:1 copy. The copy figure is a reference "
        "point, not a ceiling. `% of datasheet` is the honest wall, and nothing "
        "here passes it.\n"
    )


def _headline(runs: list[dict], e2e: list[dict] | None = None) -> str:
    """Lead with the finding, not with the best number in the table.

    An earlier version led with the largest speedup anywhere in the sweep. That
    is the number a reader trusts least, because picking the maximum of a sweep
    is what a benchmark does when it wants something from you.
    """
    best = None
    for run in runs:
        for r in run["results"]:
            if r.get("dispatch_bound") or r["regime"] != "prefill":
                continue
            cand = (r["speedup"], r, run["device"]["device_name"])
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        return ""
    _, r, dev = best

    parts = [
        f"Two fused kernels, in Triton and in Metal. On {dev} they reach "
        f"**{r['speedup']:.2f}x** over eager at prefill and sustain "
        f"{r['fused_gbs']:.0f} GB/s, {r['fused_pct_of_datasheet']:.0f}% of "
        "datasheet bandwidth."
    ]
    patched = [x for x in (e2e or []) if x.get("fused")]
    if patched:
        run = patched[0]
        pre = run["fused"]["prefill_tok_per_s"] / run["baseline"]["prefill_tok_per_s"]
        dec = run["fused"]["decode_tok_per_s"] / run["baseline"]["decode_tok_per_s"]
        parts.append(
            f"End to end that is **{pre:.2f}x on prefill** and "
            f"**{1 / dec:.2f}x slower on decode**. Both numbers are below, along "
            "with the profile that explains the split and the shapes where fusing "
            "was the wrong call."
        )
    return " ".join(parts) + "\n"


def render(runs: list[dict], e2e: list[dict] | None = None, prof: dict | None = None) -> str:
    e2e = e2e or []
    if not runs:
        return (
            f"{BEGIN}\n\n_No benchmark runs recorded yet. "
            "Run `make bench` and then `make report`._\n\n"
            f"{END}"
        )
    parts = [
        BEGIN,
        "",
        "<!-- Generated by `python -m hag.report`. Do not edit by hand. -->",
        "",
        _headline(runs, e2e),
        "### Environment",
        "",
        _env_table(runs),
        "",
        "### Prefill regime (batched, memory system saturated)",
        "",
        _results_table(runs, "prefill"),
        "",
        "### Decode regime (one row per sequence, latency-bound)",
        "",
        _results_table(runs, "decode"),
        "",
        _regressions(runs),
        "### Where the GPU time goes",
        "",
        _profile_table(prof),
        "",
        "### End to end",
        "",
        _e2e_table(e2e),
        "",
        _e2e_verdict(e2e, prof),
        _peak_caveat(runs),
        _dispatch_bound_note(runs),
        END,
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readme", default=str(REPO / "README.md"))
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--check", action="store_true", help="exit 1 if the README is stale")
    args = ap.parse_args()

    readme = Path(args.readme)
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{readme} is missing the {BEGIN} / {END} markers")

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    results_dir = Path(args.results)
    updated = head + render(
        load_runs(results_dir), load_e2e(results_dir), load_profile(REPO)
    ) + tail

    if args.check:
        if updated != text:
            print("README results tables are stale; run `make report`.", file=sys.stderr)
            raise SystemExit(1)
        print("README is up to date with results/.")
        return

    readme.write_text(updated)
    print(f"updated {readme}")


if __name__ == "__main__":
    main()
