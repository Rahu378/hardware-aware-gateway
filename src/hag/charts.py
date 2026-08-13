"""Render the results as SVG, straight from the committed JSON.

    python -m hag.charts

Three charts, one point each. Anything a table already says clearly is left to
the table; a chart earns its place only by showing a shape that numbers in a row
do not.

Every value is read from `results/` and `profiles/`, so a chart cannot claim
something the runs do not, and regenerating after a new run is a one-liner
rather than an editing session in a drawing tool.

Written to a white card rather than a transparent background: GitHub renders
README images against both light and dark page themes, and transparent SVGs
with dark text vanish in dark mode.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "img"

INK = "#1b1f24"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
RED = "#dc2626"
GREY = "#9ca3af"
GREEN = "#059669"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "figure.dpi": 110,
        }
    )
    return plt


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def chart_where_the_time_goes(plt, profile: dict, e2e: dict, graphs: dict) -> Path | None:
    """The headline: a decode token split into work and waiting, before and after.

    This is the chart that carries the project. A speedup bar would say the
    number and nothing else; splitting the bar shows that the part which shrank
    was never GPU work, which is the whole finding.
    """
    from . import roofline

    ops = _load(REPO / "results" / "ops_tesla-t4_fp16.json")
    a = roofline.analyse(profile, e2e, ops)
    if not a or not graphs:
        return None

    kernel = a["gemv_ms_per_decode_step"]
    eager_total = 1e3 / graphs["eager"]["median"]
    graph_total = 1e3 / graphs["graphed"]["median"]
    floor = a["bandwidth_floor_ms_per_token"]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    # Reversed so the "before" row reads above the "after" row.
    labels = ["CUDA graph", "eager"]
    totals = [graph_total, eager_total]
    kernels = [kernel, kernel]
    gaps = [t - kernel for t in totals]

    ax.barh(labels, kernels, color=BLUE, height=0.45, label="GPU kernel time")
    ax.barh(labels, gaps, left=kernels, color=GREY, height=0.45,
            label="CPU dispatch, GPU idle")
    ax.axvline(floor, color=RED, lw=1.4, ls="--")
    ax.text(floor + 0.5, 1.62, f"bandwidth floor {floor:.1f} ms",
            color=RED, fontsize=8.5, va="center")

    for y, (t, k) in enumerate(zip(totals, kernels, strict=True)):
        ax.text(t + 0.6, y, f"{t:.1f} ms", va="center", fontsize=10, color=INK)
        ax.text(k / 2, y, f"{k:.1f}", va="center", ha="center",
                fontsize=8.5, color="white")

    ax.set_xlim(0, eager_total * 1.2)
    ax.set_ylim(-0.6, 1.95)
    ax.set_xlabel("milliseconds per decoded token")
    ax.set_title(
        "Decode was waiting on the CPU, not the memory system\n"
        f"Qwen2.5-1.5B on a Tesla T4. Kernel time is unchanged; "
        f"{eager_total - graph_total:.1f} ms of dispatch overhead is gone.",
        fontsize=10.5, loc="left", pad=14,
    )
    # Below the axes: inside them it sat on top of the eager bar.
    ax.legend(frameon=False, fontsize=8.5, ncols=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.26))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = OUT / "where-the-time-goes.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def chart_fusion_crossover(plt, ops_runs: list[dict]) -> Path | None:
    """Speedup against row count, including where it drops below 1.

    The crossover is the shape worth seeing: fusion is not a speedup, it is a
    speedup above a threshold, and the threshold is a property of the machine.
    """
    if not ops_runs:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    colours = {"Tesla T4": BLUE, "Apple M3": GREEN}
    marks = {"swiglu": "o", "rmsnorm_residual": "s"}

    for run in ops_runs:
        dev = run["device"]["device_name"]
        for op, mark in marks.items():
            # One point per row count, median across the hidden sizes measured.
            # Plotting every width as a single series made the line zigzag
            # between three different models rather than showing the trend.
            by_rows: dict[int, list[float]] = {}
            for r in run["results"]:
                if r["op"] == op and not r.get("dispatch_bound"):
                    by_rows.setdefault(r["rows"], []).append(r["speedup"])
            pts = sorted(
                (rows, sorted(v)[len(v) // 2]) for rows, v in by_rows.items()
            )
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker=mark, ms=4.5, lw=1.6,
                    color=colours.get(dev, GREY),
                    ls="-" if op == "swiglu" else "--",
                    label=f"{dev} {op}")

    ax.axhline(1.0, color=RED, lw=1.2)
    # Right-aligned at the far end: on the left it sat on top of the M3 line.
    ax.text(2048, 1.06, "break-even: below this line, fusing is slower",
            color=RED, fontsize=8.5, va="bottom", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rows in flight  (1 = decode, 2048 = prefill)")
    ax.set_ylabel("speedup over eager")
    ax.set_title(
        "Fusion pays only once there are enough rows to be bandwidth-bound\n"
        "Below the crossover the kernel dispatches to eager instead.",
        fontsize=10.5, loc="left", pad=12,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = OUT / "fusion-crossover.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def chart_launch_floor(plt, ops_runs: list[dict]) -> Path | None:
    """The cross-platform point: dispatch cost decides what is optimisable."""
    rows = [
        (r["device"]["device_name"], r.get("dispatch_floor_ms", 0) * 1000,
         r.get("measured_copy_gbs", 0))
        for r in ops_runs
        if r.get("dispatch_floor_ms")
    ]
    if not rows:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.7))
    names = [r[0] for r in rows]
    colours = [BLUE if "T4" in n else GREEN for n in names]

    ax1.bar(names, [r[1] for r in rows], color=colours, width=0.5)
    for i, r in enumerate(rows):
        ax1.text(i, r[1], f"{r[1]:.0f} us", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("microseconds")
    ax1.set_title("Cost of one kernel launch", fontsize=10, loc="left")
    ax1.grid(axis="x", visible=False)

    ax2.bar(names, [r[2] for r in rows], color=colours, width=0.5)
    for i, r in enumerate(rows):
        ax2.text(i, r[2], f"{r[2]:.0f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("GB/s")
    ax2.set_title("Measured copy bandwidth", fontsize=10, loc="left")
    ax2.grid(axis="x", visible=False)

    fig.suptitle(
        "Same kernels, different machines: launch cost decides what is worth optimising",
        fontsize=10.5, x=0.01, ha="left", y=1.06,
    )
    fig.tight_layout()
    path = OUT / "launch-floor.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    try:
        plt = _style()
    except ImportError as exc:
        raise SystemExit("Needs matplotlib:  pip install -e '.[charts]'") from exc

    OUT.mkdir(parents=True, exist_ok=True)
    results = REPO / "results"
    ops_runs = [json.loads(p.read_text()) for p in sorted(results.glob("ops_*.json"))]
    profile = _load(REPO / "profiles" / "torch_profile_summary.json")
    e2e = _load(results / "e2e_qwen2.5-1.5b_tesla-t4.json")
    graphs = _load(results / "graphs_qwen2.5-1.5b_tesla-t4.json")

    made = [
        chart_where_the_time_goes(plt, profile, e2e, graphs),
        chart_fusion_crossover(plt, ops_runs),
        chart_launch_floor(plt, ops_runs),
    ]
    for path in made:
        print(f"wrote {path.relative_to(REPO)}" if path else "skipped (missing data)")


if __name__ == "__main__":
    main()
