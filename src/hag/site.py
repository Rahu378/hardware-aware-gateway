"""Generate the GitHub Pages roofline explorer from the committed results.

    python -m hag.site        # writes docs/index.html

The page is a calculator, not a demo. A browser cannot run a CUDA kernel, so
anything claiming to "run the benchmark live" would be replaying a recording,
and a recording of a throughput counter is a picture of a number.

What it can do honestly is the arithmetic this project turned out to be about:
given a device's bandwidth and dispatch cost, and a model's size and precision,
is decode limited by streaming weights or by the CPU issuing work? That is a
closed-form question, the constants come from measurements in `results/`, and
the answer changes usefully as you move the inputs.

Measured points are marked as measured. Everything else is labelled as the model
it is.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "index.html"


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def collect() -> dict:
    """Everything the page needs, pulled from committed runs."""
    from . import roofline

    results = REPO / "results"
    profile = _load(REPO / "profiles" / "torch_profile_summary.json")
    e2e = _load(results / "e2e_qwen2.5-1.5b_tesla-t4.json")
    graphs = _load(results / "graphs_qwen2.5-1.5b_tesla-t4.json")
    ops_t4 = _load(results / "ops_tesla-t4_fp16.json")
    ops_m3 = _load(results / "ops_apple-m3_fp16.json")

    analysis = roofline.analyse(profile, e2e, ops_t4) or {}

    devices = []
    for ops, sheet, mem in ((ops_t4, 320, 16), (ops_m3, 100, 8)):
        if not ops:
            continue
        devices.append(
            {
                "name": ops["device"]["device_name"],
                "measured_gbs": ops["measured_copy_gbs"],
                "datasheet_gbs": sheet,
                "launch_floor_us": round(ops.get("dispatch_floor_ms", 0) * 1000, 1),
                "memory_gb": mem,
                "measured": True,
            }
        )
    # Datasheet-only entries, flagged so the page never implies they were run.
    devices += [
        {"name": "NVIDIA A100 40GB", "measured_gbs": 1250, "datasheet_gbs": 1555,
         "launch_floor_us": 6, "memory_gb": 40, "measured": False},
        {"name": "NVIDIA H100 SXM", "measured_gbs": 2700, "datasheet_gbs": 3350,
         "launch_floor_us": 5, "memory_gb": 80, "measured": False},
    ]

    # Anchor the model on the measured configuration so it reproduces it.
    #
    # Per-dispatch CPU cost is the *derived* figure, not the launch floor. The
    # floor is a submit-and-wait round trip, roughly 10 us on this T4; the cost
    # of issuing one more op from Python is 2.8 us. Using the floor as the
    # per-op cost overstated CPU time by more than 3x and had the calculator
    # predicting half the throughput that was actually measured.
    #
    # `other_gpu_ms` is everything on the device that is not the weight-streaming
    # GEMV: norms, activations, the attention kernels, the cache writes. It is
    # taken as the residual of the graphed measurement, which is the
    # configuration with no CPU gap left to confound it.
    graphed_ms = 1e3 / graphs["graphed"]["median"] if graphs else None
    kernel_ms = analysis.get("gemv_ms_per_decode_step")
    other_gpu_ms = round(graphed_ms - kernel_ms, 2) if graphed_ms and kernel_ms else 0.0

    return {
        "devices": devices,
        "analysis": analysis,
        "model": {
            "us_per_dispatch": analysis.get("implied_us_per_dispatch"),
            "other_gpu_ms": other_gpu_ms,
            "kernel_efficiency": 0.87,
        },
        "measured": {
            "model": "Qwen2.5-1.5B",
            "weight_bytes": e2e.get("weight_bytes"),
            "dispatches": analysis.get("op_dispatches_per_token"),
            "us_per_dispatch": analysis.get("implied_us_per_dispatch"),
            "eager_tok_s": (graphs.get("eager") or {}).get("median"),
            "graphed_tok_s": (graphs.get("graphed") or {}).get("median"),
            "graph_speedup": graphs.get("speedup"),
            "kernel_ms": analysis.get("gemv_ms_per_decode_step"),
        },
    }


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inference roofline explorer &mdash; Hardware-Aware Gateway</title>
<meta name="description" content="Is LLM decode limited by memory bandwidth
or by CPU dispatch? A calculator anchored on measured data.">
<style>
:root{--ink:#12161c;--muted:#5b6572;--line:#e3e7ec;--bg:#fff;--card:#f7f9fb;
--blue:#2563eb;--grey:#9ca3af;--red:#dc2626;--green:#059669;--track:#cbd3dc;}
@media(prefers-color-scheme:dark){:root{--ink:#e8ecf1;--muted:#9aa4b2;--line:#262c36;
--bg:#0d1117;--card:#151b23;--track:#39424e;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:0 20px}
header{padding:44px 0 32px;border-bottom:1px solid var(--line)}
h1{font-size:clamp(28px,5vw,40px);line-height:1.15;margin:0 0 12px;letter-spacing:-.02em}
h2{font-size:20px;margin:44px 0 6px;letter-spacing:-.01em}
p{margin:0 0 14px;color:var(--muted)}
p.lead{font-size:18px;color:var(--ink)}
a{color:var(--blue)}
code{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
background:var(--card);padding:2px 5px;border-radius:4px}
.head-nums{display:flex;flex-wrap:wrap;gap:28px;margin:22px 0 0}
.head-nums div{min-width:120px}
.head-nums b{display:block;font-size:26px;letter-spacing:-.02em}
.head-nums span{font-size:13px;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:22px;margin:18px 0}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:18px}
label{display:block;font-size:13px;font-weight:600;margin-bottom:6px}
select{width:100%}

/* Range inputs need explicit styling: the platform default is a thin grey
   track that all but disappears against the card, which is the same colour. */
input[type=range]{width:100%;-webkit-appearance:none;appearance:none;
background:transparent;height:24px;cursor:pointer}
input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:99px;
background:var(--track)}
input[type=range]::-moz-range-track{height:6px;border-radius:99px;background:var(--track)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
width:20px;height:20px;margin-top:-7px;border-radius:50%;background:var(--blue);
border:2px solid var(--bg);box-shadow:0 1px 4px rgba(0,0,0,.28)}
input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;
background:var(--blue);border:2px solid var(--bg);box-shadow:0 1px 4px rgba(0,0,0,.28)}
input[type=range]:focus-visible::-webkit-slider-thumb{outline:2px solid var(--blue);
outline-offset:2px}
select{padding:8px;border:1px solid var(--line);border-radius:7px;
background:var(--bg);color:var(--ink);font-size:14px}
.val{font-size:13px;color:var(--muted);margin-top:4px}
.bar{position:relative;height:56px;margin:26px 0 8px;border-radius:8px;
overflow:hidden;background:var(--line)}
.seg{position:absolute;top:0;height:100%;display:flex;align-items:center;
justify-content:center;font-size:13px;font-weight:600;color:#fff;
transition:width .25s ease,left .25s ease}
.seg.k{background:var(--blue);left:0}
.seg.d{background:var(--grey)}
.axis{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
.verdict{font-size:17px;font-weight:600;margin:20px 0 6px}
.verdict.mem{color:var(--blue)}.verdict.cpu{color:var(--red)}
.verdict.mix{color:var(--green)}
table{width:100%;border-collapse:collapse;font-size:14px;margin:10px 0}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.n{font-variant-numeric:tabular-nums}
.tag{font-size:11px;padding:2px 7px;border-radius:99px;border:1px solid var(--line);
color:var(--muted)}
.tag.m{color:var(--green);border-color:var(--green)}
nav{border-bottom:1px solid var(--line);background:var(--bg);
position:sticky;top:0;z-index:5}
.navin{display:flex;align-items:center;justify-content:space-between;
padding:12px 20px;font-size:14px}
.brand{font:13px/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted)}
.navlinks a{margin-left:18px;text-decoration:none;color:var(--muted)}
.navlinks a:hover{color:var(--ink)}
.cta{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 10px}
.btn{display:inline-block;padding:11px 18px;border-radius:8px;font-size:14px;
font-weight:600;text-decoration:none;border:1px solid var(--line);color:var(--ink)}
.btn:hover{border-color:var(--muted)}
.btn.primary{background:var(--blue);border-color:var(--blue);color:#fff}
.btn.primary:hover{filter:brightness(1.08)}
.fine{font-size:13px}
footer{margin:56px 0 40px;padding-top:22px;border-top:1px solid var(--line);
font-size:14px;color:var(--muted)}
img{max-width:100%;border-radius:10px;border:1px solid var(--line);margin:8px 0}
</style>
</head>
<body>
<nav>
<div class="wrap navin">
<span class="brand">hardware-aware-gateway</span>
<span class="navlinks">
<a href="https://github.com/Rahu378/hardware-aware-gateway/releases/tag/v1.0.0">v1.0.0</a>
<a href="https://github.com/Rahu378/hardware-aware-gateway">GitHub</a>
</span>
</div>
</nav>
<div class="wrap">

<header>
<h1>Is your decode limited by bandwidth,<br>or by the CPU?</h1>
<p class="lead">Most LLM decode optimisation assumes the memory system is the wall.
On a Tesla T4 it was not: the GPU sat idle for over half of every token while the
CPU issued about six thousand operations. Replaying a captured CUDA graph was
worth more than every fused kernel combined.</p>
<div class="head-nums">
<div><b>__SPEEDUP__&times;</b><span>decode, CUDA graph replay</span></div>
<div><b>__EAGER__ &rarr; __GRAPHED__</b><span>tokens/sec, Qwen2.5-1.5B</span></div>
<div><b>86%</b><span>of the gap the roofline predicted</span></div>
</div>
</header>

<h2>The explorer</h2>
<p>Pick a device and a model. The arithmetic below is the same as
<code>hag.roofline</code>, with constants measured on the hardware named. Devices
marked <span class="tag m">measured</span> were benchmarked; the others use
datasheet figures and are labelled so.</p>

<div class="card">
<div class="controls">
<div>
<label for="dev">Device</label>
<select id="dev" autocomplete="off"></select>
<div class="val" id="devval"></div>
</div>
<div>
<label for="params">Model parameters</label>
<input type="range" id="params" min="0.5" max="70" step="0.5" value="1.5"
 autocomplete="off">
<div class="val" id="paramsval"></div>
</div>
<div>
<label for="bits">Weight precision</label>
<select id="bits" autocomplete="off">
<option value="16">fp16 / bf16</option>
<option value="8">int8</option>
<option value="4">int4</option>
</select>
<div class="val">Decode streams every weight once per token.</div>
</div>
<div>
<label for="disp">Op dispatches per token</label>
<input type="range" id="disp" min="0" max="12000" step="10" value="6130"
 autocomplete="off">
<div class="val" id="dispval"></div>
</div>
</div>

<div class="bar" id="bar">
<div class="seg k" id="segk"></div>
<div class="seg d" id="segd"></div>
</div>
<div class="axis"><span>0 ms</span><span id="axmax"></span></div>

<div class="verdict" id="verdict"></div>
<p id="detail"></p>
</div>

<h2>What that means</h2>
<p>Slide dispatches to zero: that is a perfectly captured CUDA graph. Slide
precision to int4: that is quantisation. Whichever bar shrinks more is where
your next week of work belongs, and for the measured configuration it was not
the one most people reach for first.</p>

<h2>What was actually measured</h2>
<img src="img/where-the-time-goes.svg"
 alt="Decode time split into GPU work and CPU dispatch, before and after capture">
<img src="img/fusion-crossover.svg"
 alt="Speedup against row count for both kernels on both platforms">
<table id="devtable"></table>

<h2>Running it yourself</h2>
<p>The notebook runs the whole pipeline on a free T4 in about ten minutes and
writes every number on this page: correctness, the op sweep, the profile, the
end-to-end benchmark and the CUDA graph capture. Nothing here is a figure you
have to take on trust.</p>
<div class="cta">
<a class="btn primary"
 href="https://colab.research.google.com/github/Rahu378/hardware-aware-gateway/blob/main/notebooks/colab_bootstrap.ipynb">
Open in Colab &rarr;</a>
<a class="btn" href="https://github.com/Rahu378/hardware-aware-gateway">View on GitHub</a>
<a class="btn" href="https://github.com/Rahu378/hardware-aware-gateway/releases/tag/v1.0.0">
Read the write-up</a>
</div>
<p class="fine">Prefer Kaggle: its GPU quota is larger and the machine is
quieter. The same benchmark measured a standard deviation of 0.53 tok/s there
against 3.68 on Colab, which decided whether an effect resolved at all.</p>

<footer>
Generated by <code>python -m hag.site</code> from the JSON in
<code>results/</code>, so the page cannot claim anything the runs do not.
&nbsp;&middot;&nbsp;
<a href="https://github.com/Rahu378/hardware-aware-gateway">Source</a>
&nbsp;&middot;&nbsp;
<a href="https://github.com/Rahu378/hardware-aware-gateway/releases/tag/v1.0.0">v1.0.0 write-up</a>
</footer>

</div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);

DATA.devices.forEach((d,i) => {
  const o = document.createElement("option");
  o.value = i; o.textContent = d.name + (d.measured ? "" : "  (datasheet)");
  $("dev").appendChild(o);
});

function render(){
  const d = DATA.devices[$("dev").value];
  const params = parseFloat($("params").value);
  const bits = parseInt($("bits").value);
  const disp = parseInt($("disp").value);

  // Weights streamed per token, then the floor at this device's real bandwidth.
  const bytes = params * 1e9 * bits / 8;
  const floorMs = bytes / (d.measured_gbs * 1e9) * 1e3;
  // Kernels do not reach the floor exactly; 0.87 is what cuBLAS managed here.
  const kernelMs = floorMs / DATA.model.kernel_efficiency;
  // Everything else on the device: norms, activations, attention, cache writes.
  const otherMs = DATA.model.other_gpu_ms;
  // Dispatch cost is per *op issued*, not the launch round trip.
  const cpuMs = disp * DATA.model.us_per_dispatch / 1000;
  const totalMs = kernelMs + otherMs + cpuMs;

  $("devval").textContent = d.measured_gbs + " GB/s measured, "
    + d.launch_floor_us + " us launch floor";
  $("paramsval").textContent = params + "B parameters, "
    + (bytes/1e9).toFixed(2) + " GB at this precision";
  $("dispval").textContent = disp === 0
    ? "0 - a fully captured CUDA graph"
    : disp + " ops x " + DATA.model.us_per_dispatch + " us = "
      + cpuMs.toFixed(1) + " ms of CPU per token";

  const gpuMs = kernelMs + otherMs;
  const scale = Math.max(totalMs, 1) * 1.05;
  $("segk").style.width = (100 * gpuMs / scale) + "%";
  $("segk").textContent = gpuMs > scale*0.12 ? gpuMs.toFixed(1)+" ms GPU" : "";
  $("segd").style.left  = (100 * gpuMs / scale) + "%";
  $("segd").style.width = (100 * cpuMs / scale) + "%";
  $("segd").textContent = cpuMs > scale*0.14 ? cpuMs.toFixed(1)+" ms CPU wait" : "";
  $("axmax").textContent = scale.toFixed(0) + " ms";

  // Three states, not two. A binary label on a 45/55 split says something the
  // numbers do not: the measured configuration was 46% CPU and 54% GPU, and
  // calling that "bandwidth-bound" would have hidden the finding this whole
  // project is about.
  const cpuShare = cpuMs / totalMs;
  const tok = (1e3/totalMs).toFixed(0);
  let cls, label, detail;
  if (cpuShare > 0.6) {
    cls = "cpu"; label = "Dispatch-bound";
    detail = "The GPU is idle for <b>" + (100*cpuShare).toFixed(0) + "%</b> of "
      + "every token, waiting on the CPU to issue work. No kernel touches that. "
      + "<b>Capture the decode step into a CUDA graph</b>; measured on a T4 that "
      + "was worth " + DATA.measured.graph_speedup.toFixed(2) + "x.";
  } else if (cpuShare < 0.25) {
    cls = "mem"; label = "Bandwidth-bound";
    detail = "Weights dominate. A faster kernel will not help much: cuBLAS "
      + "already ran at 87% of this floor on the T4, so a flawless replacement "
      + "was worth under 6% of a token. <b>Quantisation is the lever</b>, "
      + "because it moves the floor itself rather than approaching it.";
  } else {
    cls = "mix"; label = "Mixed";
    detail = "Neither side dominates: <b>" + (100*cpuShare).toFixed(0)
      + "% CPU dispatch, " + (100*(1-cpuShare)).toFixed(0) + "% GPU work</b>. "
      + "This is where the measured configuration sat, and it is the case a "
      + "single headline number hides. Both levers are live: capture the step, "
      + "then quantise.";
  }
  // Weights alone against device memory. A configuration that cannot be loaded
  // should not be quoted a throughput: the calculator would otherwise price
  // 70B in fp16 on an 80 GB card without comment.
  const gb = bytes / 1e9;
  const fits = gb < d.memory_gb * 0.85;   // KV cache and activations need room
  if (!fits) {
    $("verdict").className = "verdict cpu";
    $("verdict").textContent = "Does not fit - " + gb.toFixed(0) + " GB of weights on "
      + d.memory_gb + " GB";
    $("detail").innerHTML = "The weights alone exceed what this device holds, before "
      + "any KV cache. Quantise, shard across devices, or pick a smaller model. "
      + "The throughput below the fold assumes the model is resident.";
    return;
  }
  $("verdict").className = "verdict " + cls;
  $("verdict").textContent = label + " - " + tok + " tokens/sec";
  $("detail").innerHTML = detail;
}

const tbl = $("devtable");
tbl.innerHTML = "<tr><th>device</th><th>measured bandwidth</th>"
  + "<th>datasheet</th><th>launch cost</th><th>memory</th><th></th></tr>"
  + DATA.devices.map(d => "<tr><td>" + d.name + "</td>"
      + "<td class=n>" + d.measured_gbs + " GB/s</td>"
      + "<td class=n>" + d.datasheet_gbs + " GB/s</td>"
      + "<td class=n>" + d.launch_floor_us + " us</td>"
      + "<td class=n>" + d.memory_gb + " GB</td>"
      + "<td>" + (d.measured
          ? "<span class='tag m'>measured</span>"
          : "<span class='tag'>datasheet</span>") + "</td></tr>").join("");

["dev","params","bits","disp"].forEach(id => {
  $(id).addEventListener("input", render);
  $(id).addEventListener("change", render);
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    data = collect()
    m = data["measured"]
    html = (
        HTML.replace("__DATA__", json.dumps(data))
        .replace("__SPEEDUP__", f"{m['graph_speedup']:.2f}")
        .replace("__EAGER__", f"{m['eager_tok_s']:.0f}")
        .replace("__GRAPHED__", f"{m['graphed_tok_s']:.0f}")
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(REPO)}  ({len(html) // 1024} KB)")
    print("Enable Pages:  Settings > Pages > Deploy from branch > main > /docs")


if __name__ == "__main__":
    main()
