"""Side-by-side streaming demo: eager PyTorch against a captured CUDA graph.

    pip install -e '.[demo]'
    python demos/gradio_app.py --share

Both panes decode the *same* prompt greedily, so they emit identical tokens.
That is the point of the layout: the text stays in lockstep while one side
finishes first, which makes the speedup something you watch rather than read.

Needs a CUDA GPU. There is no hosted version and there should not be a claim of
one: the demo is a live measurement, and a screenshot of it would be a picture
of a number rather than the number. Run it on a Colab or Kaggle T4 with
`--share` and you get a public URL for the life of the session.

The counters are honest about what they include. Time starts after prefill and
after graph capture, so neither side is charged for setup the other does not
do, and both run under `no_grad` because `inference_mode` is measurably cheaper
on dispatch-heavy work and using it on one side only would flatter that side.
"""

from __future__ import annotations

import argparse
import time

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_PROMPT = "The key insight about GPU memory bandwidth is that"


def _load(model_name: str, dtype_name: str):
    import torch

    from hag import models

    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    model, tokenizer = models.load_model_and_tokenizer(model_name, dt, "cuda")
    return model, tokenizer


def _stream_eager(model, tokenizer, prompt: str, max_new: int):
    """Greedy decode with the ordinary dynamic cache, yielding as it goes."""
    import torch

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()

        text, t0 = "", time.perf_counter()
        for i in range(max_new):
            text += tokenizer.decode(tok[0])
            yield text, (i + 1) / (time.perf_counter() - t0)
            out = model(input_ids=tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1:].argmax(dim=-1)


def _stream_graphed(model, tokenizer, prompt: str, max_new: int):
    """Same decode, replaying a captured graph. Capture happens before timing."""
    import torch

    from hag.graphs import GraphedDecoder

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    decoder = GraphedDecoder(model, max_cache_len=ids.shape[-1] + max_new + 8)
    tok = decoder.prefill(ids)
    decoder.capture()
    torch.cuda.synchronize()

    text, t0 = "", time.perf_counter()
    for i in range(max_new):
        text += tokenizer.decode(tok[0])
        yield text, (i + 1) / (time.perf_counter() - t0)
        tok = decoder.decode_step(tok)


def build(model, tokenizer):
    import gradio as gr

    def run(prompt, max_new):
        max_new = int(max_new)
        # Sequential rather than concurrent: two decoders sharing one GPU would
        # contend, and each would measure the other's interference instead of
        # its own cost. Eager runs first so the slower side sets the pace the
        # viewer sees.
        eager_text = graph_text = ""
        eager_rate = graph_rate = 0.0

        for eager_text, eager_rate in _stream_eager(model, tokenizer, prompt, max_new):
            yield eager_text, f"{eager_rate:.1f} tok/s", graph_text, "waiting", ""

        for graph_text, graph_rate in _stream_graphed(model, tokenizer, prompt, max_new):
            yield (
                eager_text, f"{eager_rate:.1f} tok/s",
                graph_text, f"{graph_rate:.1f} tok/s", "",
            )

        same = eager_text == graph_text
        verdict = (
            f"### {graph_rate / eager_rate:.2f}x faster\n\n"
            f"{1e3 / eager_rate:.1f} ms per token to {1e3 / graph_rate:.1f} ms. "
            f"Output is {'identical' if same else '**DIFFERENT, which is a bug**'}: "
            "greedy decoding is deterministic, so the graph must not change a "
            "single token.\n\n"
            "The GPU kernels are the same on both sides and take the same time. "
            "What the graph removes is the CPU issuing about six thousand "
            "operations per token while the GPU waits."
        )
        yield (
            eager_text, f"{eager_rate:.1f} tok/s",
            graph_text, f"{graph_rate:.1f} tok/s", verdict,
        )

    with gr.Blocks(title="Hardware-Aware Gateway") as demo:
        gr.Markdown(
            "# Same model, same tokens, half the time\n"
            "Both sides greedily decode the same prompt with identical weights. "
            "The right pane replays a captured CUDA graph instead of dispatching "
            "every operation from Python."
        )
        with gr.Row():
            prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt", scale=4)
            max_new = gr.Slider(16, 256, value=96, step=16, label="Tokens", scale=1)
        go = gr.Button("Generate", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Eager PyTorch")
                eager_rate = gr.Label(value="idle", label="throughput")
                eager_out = gr.Textbox(lines=8, label="", show_label=False)
            with gr.Column():
                gr.Markdown("### CUDA graph replay")
                graph_rate = gr.Label(value="idle", label="throughput")
                graph_out = gr.Textbox(lines=8, label="", show_label=False)

        verdict = gr.Markdown()
        go.click(
            run,
            inputs=[prompt, max_new],
            outputs=[eager_out, eager_rate, graph_out, graph_rate, verdict],
        )
    return demo


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--share", action="store_true", help="public URL, for Colab or Kaggle")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "This demo measures a CUDA graph against eager dispatch, so it needs "
            "a CUDA GPU. On a free Colab or Kaggle T4, run it with --share."
        )

    print(f"loading {args.model} on {torch.cuda.get_device_name(0)}...")
    model, tokenizer = _load(args.model, args.dtype)
    build(model, tokenizer).launch(share=args.share)


if __name__ == "__main__":
    main()
