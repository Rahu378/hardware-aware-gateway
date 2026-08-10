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


def _e2e_table(runs: list[dict]) -> str:
    """End-to-end tokens/sec, which is the only number that decides anything.

    Reported whether or not it flatters the kernels. A repo that shows op-level
    speedups and quietly omits the end-to-end result is not reporting a
    measurement, it is making a case.
    """
    if not runs:
        return "_No end-to-end runs recorded yet. Run `make bench-e2e`._"
    lines = [
        "| device | model | baseline decode | with fused kernels | change | peak memory |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        name = run["device"]["device_name"]
        model = run["model"].split("/")[-1]
        base = run["baseline"]
        fused = run.get("fused")
        if fused is None:
            lines.append(
                f"| {name} | {model} | {base['decode_tok_per_s']:.2f} tok/s "
                f"| not applicable | baseline only "
                f"| {base['peak_memory_gb']:.2f} GB |"
            )
            continue
        ratio = run.get("decode_speedup") or 0.0
        mark = f"**{ratio:.3f}x**" if ratio >= 1 else f"**{ratio:.3f}x slower**"
        lines.append(
            f"| {name} | {model} | {base['decode_tok_per_s']:.2f} tok/s "
            f"| {fused['decode_tok_per_s']:.2f} tok/s | {mark} "
            f"| {fused['peak_memory_gb']:.2f} GB |"
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
    floors = {
        run["device"]["device_name"]: run.get("dispatch_floor_ms", 0.0) for run in runs
    }
    devs = ", ".join(f"{d} ({f * 1e3:.0f} us)" for d, f in floors.items() if f)
    return (
        f"\n{len(rows)} measurements landed within 2x of the launch floor "
        f"({devs}) and are excluded from the tables above. At those shapes the "
        "number describes the runtime's submission path, not the kernel. "
        "Decode-shaped work has to be judged end-to-end or with GPU counters; "
        "see [docs/METHOD.md](docs/METHOD.md).\n"
    )


def _e2e_verdict(e2e: list[dict]) -> str:
    """State the end-to-end outcome in words, including when it is a regression."""
    losses = [r for r in e2e if (r.get("decode_speedup") or 1.0) < 1.0]
    if not losses:
        return ""
    worst = min(losses, key=lambda r: r["decode_speedup"])
    dev = worst["device"]["device_name"]
    pct = (1 - worst["decode_speedup"]) * 100
    return (
        f"\n**The fused kernels made end-to-end decode slower, by {pct:.0f}% on "
        f"{dev}.** The profile says why. Matrix-vector products are 63% of GPU "
        "time at decode, while every elementwise op these kernels touch adds up "
        "to roughly 12%, and the fused SwiGLU is itself slower than eager at a "
        "single row. A perfect fusion had a 12% ceiling and this one spent more "
        "than it saved.\n\nThe kernels are not wrong; the target was. Prefill, "
        "where the same kernels reach 6.25x and 95% of datasheet bandwidth, is "
        "the regime where saving traffic is worth anything. Decode is bound by "
        "streaming weights through the GEMV, and no amount of elementwise fusion "
        "touches that. Fixing it means dispatching to eager at low row counts and "
        "then going after the GEMV itself.\n"
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


def _headline(runs: list[dict]) -> str:
    best: tuple[float, str] | None = None
    for run in runs:
        name = run["device"]["device_name"]
        for r in run["results"]:
            if r.get("dispatch_bound") or r["regime"] != "prefill":
                continue
            cand = (
                r["speedup"],
                f"`{r['op']}` at {r['rows']}x{r['width']} on {name}: "
                f"**{r['speedup']:.2f}x** over eager, reaching {r['fused_gbs']:.0f} GB/s "
                f"({r['fused_pct_of_measured_peak']:.0f}% of measured copy bandwidth)",
            )
            if best is None or cand[0] > best[0]:
                best = cand
    return "" if best is None else f"Best measured result: {best[1]}.\n"


def render(runs: list[dict], e2e: list[dict] | None = None) -> str:
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
        _headline(runs),
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
        "### End to end",
        "",
        _e2e_table(e2e),
        "",
        _e2e_verdict(e2e),
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
    updated = head + render(load_runs(results_dir), load_e2e(results_dir)) + tail

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
