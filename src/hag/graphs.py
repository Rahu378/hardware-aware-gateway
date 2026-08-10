"""CUDA graph capture for the decode step.

The roofline said decode on a T4 is dispatch-bound, not bandwidth-bound: about
6000 op dispatches per token at roughly 3 us each, against 14.7 ms of actual
kernel time in a 32.5 ms token. No kernel addresses that. Replaying a captured
graph does, because the whole step becomes one submission.

Three things have to be true before a graph can be captured, and each one is a
way this fails:

* **Every tensor the step touches must live at a fixed address.** A KV cache
  that grows by `torch.cat` allocates a new buffer per token, so the graph
  would replay against memory that no longer holds the cache. `StaticCache`
  preallocates and writes in place, which is why it is mandatory here.
* **No control flow may depend on values the CPU has to read.** Anything that
  calls `.item()` or branches on a device tensor forces a synchronisation and
  cannot be captured.
* **Inputs must be written into the same buffers every step.** Replay does not
  take arguments. `static_input_ids.copy_(tok)` is the entire calling
  convention.

The failure mode is silent and severe: a graph that replays against stale
buffers produces fluent, wrong tokens. `verify()` exists because of that, and
the benchmark refuses to report a speedup until it passes.
"""

from __future__ import annotations

from dataclasses import dataclass


def _make_static_cache(model, max_cache_len: int, device, dtype):
    """Build a `StaticCache` across the constructor signatures transformers has used.

    The keyword moved between `max_batch_size` and `batch_size` and back over
    the 4.x series, and the class moved module in 5.x. Trying signatures is
    uglier than pinning a version and considerably more likely to work on
    whatever Colab happens to ship.
    """
    try:
        from transformers import StaticCache
    except ImportError:  # pragma: no cover
        try:
            from transformers.cache_utils import StaticCache
        except ImportError as exc:
            raise RuntimeError(
                "This transformers build has no StaticCache, so the decode step "
                "cannot be captured. A dynamically grown cache reallocates every "
                "token and a graph would replay against freed memory."
            ) from exc

    common = {"config": model.config, "max_cache_len": max_cache_len,
              "device": device, "dtype": dtype}
    errors = []
    for extra in ({"max_batch_size": 1}, {"batch_size": 1}, {}):
        try:
            return StaticCache(**common, **extra)
        except TypeError as exc:
            errors.append(f"{extra or 'no batch kwarg'}: {exc}")
    raise RuntimeError(
        "Could not construct StaticCache. Signatures tried:\n  " + "\n  ".join(errors)
    )


@dataclass
class GraphStats:
    captured: bool
    max_cache_len: int
    warmup_steps: int


class GraphedDecoder:
    """Runs prefill eagerly, then replays a captured graph for each decode step.

    Prefill is deliberately not captured. It runs once per request at a shape
    that varies with prompt length, so a graph would have to be recaptured
    constantly, and prefill is compute-bound anyway: it has no dispatch problem
    to solve.
    """

    def __init__(self, model, max_cache_len: int, device: str = "cuda", warmup_steps: int = 3):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require a CUDA device")

        self.model = model
        self.device = device
        self.max_cache_len = max_cache_len
        self.warmup_steps = warmup_steps
        self.dtype = next(model.parameters()).dtype

        self.cache = _make_static_cache(model, max_cache_len, device, self.dtype)
        self._graph = None
        self._static_out = None
        # Set here as well as in prefill(), so calling capture() out of order
        # raises something meaningful instead of AttributeError.
        self._next_position = 0

        # The only inputs the replayed step reads. Written in place each token.
        self.static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.static_cache_position = torch.zeros((1,), dtype=torch.long, device=device)

    # -- eager paths ------------------------------------------------------

    def _forward(self):
        return self.model(
            input_ids=self.static_input_ids,
            cache_position=self.static_cache_position,
            past_key_values=self.cache,
            use_cache=True,
        )

    def prefill(self, input_ids):
        """Run the prompt eagerly, filling the static cache. Returns the next token."""
        import torch

        n = input_ids.shape[-1]
        if n >= self.max_cache_len:
            raise ValueError(
                f"prompt of {n} tokens does not fit a cache of {self.max_cache_len}"
            )
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                cache_position=torch.arange(n, device=self.device),
                past_key_values=self.cache,
                use_cache=True,
            )
        self._next_position = n
        return out.logits[:, -1:].argmax(dim=-1)

    # -- capture and replay ----------------------------------------------

    def capture(self) -> GraphStats:
        """Warm up on a side stream, then capture one decode step.

        `no_grad` rather than `inference_mode`: inference tensors carry extra
        restrictions on aliasing and versioning, and the graph's output buffer is
        read again on every replay for the life of the decoder. `no_grad` is what
        the CUDA graph recipe in the PyTorch docs uses, and it is the one that
        survives being read repeatedly.

        The side-stream warmup is not optional. Capture records allocations as
        well as kernels, so any lazy initialisation that happens on the first
        call would be baked into the graph. Running it a few times first moves
        that work out of the recording.
        """
        import torch

        self.static_input_ids.fill_(0)
        self.static_cache_position.fill_(self._next_position)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(self.warmup_steps):
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph), torch.no_grad():
            self._static_out = self._forward()

        return GraphStats(True, self.max_cache_len, self.warmup_steps)

    def decode_step(self, token):
        """Replay the graph for one token. Returns the next token.

        `copy_` rather than assignment: rebinding the attribute would leave the
        graph replaying against the old buffer, which is the silent-corruption
        failure this module warns about.
        """
        if self._graph is None:
            raise RuntimeError("capture() must run before decode_step()")

        self.static_input_ids.copy_(token)
        self.static_cache_position.fill_(self._next_position)
        self._graph.replay()
        self._next_position += 1
        return self._static_out.logits[:, -1:].argmax(dim=-1)

    def generate(self, input_ids, new_tokens: int) -> list[int]:
        tok = self.prefill(input_ids)
        self.capture()
        out = [int(tok)]
        for _ in range(new_tokens - 1):
            tok = self.decode_step(tok)
            out.append(int(tok))
        return out

    def reset(self):
        """Clear the cache so another sequence can reuse the captured graph."""
        if hasattr(self.cache, "reset"):
            self.cache.reset()
        self._next_position = 0


def eager_generate(model, input_ids, new_tokens: int, device: str = "cuda") -> list[int]:
    """Greedy decode with the ordinary dynamic cache, as the reference."""
    import torch

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(dim=-1)
        toks = [int(tok)]
        for _ in range(new_tokens - 1):
            out = model(input_ids=tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1:].argmax(dim=-1)
            toks.append(int(tok))
    return toks


def static_eager_generate(model, input_ids, new_tokens: int, device: str = "cuda") -> list[int]:
    """Greedy decode with a StaticCache but no graph.

    This is the control. Going from a dynamic cache to a graphed static one
    changes two things at once, and only one of them is the graph.
    """
    decoder = GraphedDecoder(
        model, max_cache_len=input_ids.shape[-1] + new_tokens + 8, device=device
    )
    tok = decoder.prefill(input_ids)
    toks = [int(tok)]
    for _ in range(new_tokens - 1):
        with __import__("torch").no_grad():
            decoder.static_input_ids.copy_(tok)
            decoder.static_cache_position.fill_(decoder._next_position)
            out = decoder._forward()
        decoder._next_position += 1
        tok = out.logits[:, -1:].argmax(dim=-1)
        toks.append(int(tok))
    return toks


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)


def verify(model, input_ids, new_tokens: int = 16, device: str = "cuda") -> dict:
    """Does the graph change the output? Isolated from the cache change.

    Two comparisons, because switching to a graphed static cache changes two
    independent things:

    * **graphed vs static-eager** must match exactly. Same cache, same kernels,
      the only difference is replay. Greedy decoding is deterministic, so any
      divergence here is a real bug. This is the check that matters, and it is
      the one that catches a graph replaying against a buffer nobody updated,
      which otherwise produces fluent, wrong text.
    * **static-eager vs dynamic-eager** is informational. A padded static cache
      changes the shapes attention sees, so SDPA can pick different kernels and
      the last bits can differ; a token can then flip on a near-tie in the
      argmax. That is not the graph's fault, and folding it into a pass/fail
      would make the check cry wolf.
    """
    graphed_decoder = GraphedDecoder(
        model, max_cache_len=input_ids.shape[-1] + new_tokens + 8, device=device
    )
    graphed = graphed_decoder.generate(input_ids, new_tokens)
    static_eager = static_eager_generate(model, input_ids, new_tokens, device)
    dynamic_eager = eager_generate(model, input_ids, new_tokens, device)

    graph_div = _first_divergence(static_eager, graphed)
    cache_div = _first_divergence(dynamic_eager, static_eager)
    return {
        "match": graph_div is None,
        "tokens": new_tokens,
        "graph_first_divergence": graph_div,
        "static_cache_first_divergence": cache_div,
        "graphed_head": graphed[:8],
        "static_eager_head": static_eager[:8],
        "dynamic_eager_head": dynamic_eager[:8],
    }


def main() -> None:
    """Verify the graphed decoder, then benchmark it against eager.

        python -m hag.graphs --model Qwen/Qwen2.5-1.5B

    Verification runs first and a failure is fatal. Benchmarking a decoder that
    emits different tokens than the reference would be measuring the wrong
    program.
    """
    import argparse
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    from . import devices, models
    from .bench_e2e import _paired_stats, _summarise

    ap = argparse.ArgumentParser(description="CUDA graph decode: verify then benchmark")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--verify-tokens", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA graphs need a CUDA device. This is the one part of the project "
            "with no Apple silicon equivalent: Metal has no comparable capture API "
            "exposed through MLX."
        )

    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    info = devices.describe("cuda")
    print(f"device : {info['device_name']}")
    print(f"model  : {args.model} ({args.dtype})\n")

    model, tokenizer = models.load_model_and_tokenizer(args.model, dt, "cuda")
    vocab = models.vocab_size(tokenizer, model)
    input_ids = torch.randint(0, vocab, (1, args.prompt_tokens), device="cuda", dtype=torch.long)

    print("verifying graphed decode against eager...")
    check = verify(model, input_ids, args.verify_tokens)
    if not check["match"]:
        print(json.dumps(check, indent=2))
        raise SystemExit(
            f"\nGraphed and eager decode diverged at token "
            f"{check['first_divergence_index']}. Greedy decoding is deterministic, "
            "so this is a real bug, not tolerance. The usual cause is a buffer the "
            "graph does not own being rebound instead of written in place."
        )
    print(f"  match: first {args.verify_tokens} tokens identical\n")

    cache_len = args.prompt_tokens + args.new_tokens + 8

    def run_graphed() -> float:
        decoder = GraphedDecoder(model, max_cache_len=cache_len)
        tok = decoder.prefill(input_ids)
        decoder.capture()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.new_tokens):
            tok = decoder.decode_step(tok)
        torch.cuda.synchronize()
        return args.new_tokens / (time.perf_counter() - t0)

    def run_eager() -> float:
        # no_grad, matching the graphed path. inference_mode is cheaper than
        # no_grad on dispatch-heavy work, which is precisely this workload, so
        # letting the baseline use it while the graph uses no_grad would bias
        # the comparison. Same context on both sides isolates the graph.
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.new_tokens):
                out = model(input_ids=tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
                tok = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()
        return args.new_tokens / (time.perf_counter() - t0)

    # Alternated for the same reason as bench_e2e: a shared GPU drifts, and
    # measuring the two configurations in sequence puts all of that drift on
    # whichever ran second.
    eager_s, graph_s = [], []
    for i in range(args.repeats):
        eager_s.append({"tok_per_s": run_eager()})
        graph_s.append({"tok_per_s": run_graphed()})
        print(f"  repeat {i + 1}/{args.repeats}: eager {eager_s[-1]['tok_per_s']:.2f}, "
              f"graphed {graph_s[-1]['tok_per_s']:.2f} tok/s")

    es, gs = _summarise(eager_s, "tok_per_s"), _summarise(graph_s, "tok_per_s")
    paired = _paired_stats(eager_s, graph_s, "tok_per_s")
    speedup = gs["median"] / es["median"]

    print(f"\neager   : {es['median']:.2f} tok/s (range {es['min']:.2f} to {es['max']:.2f})")
    print(f"graphed : {gs['median']:.2f} tok/s (range {gs['min']:.2f} to {gs['max']:.2f})")
    print(f"speedup : {speedup:.3f}x")
    print(f"paired  : {paired['mean_diff']:+.2f} tok/s, t = {paired['t']:.2f} "
          f"against {paired['t_crit_95']:.2f}")
    print("RESOLVED" if paired["resolved"] else "NOT RESOLVED at this repeat count")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": info,
        "model": args.model,
        "dtype": args.dtype,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "repeats": args.repeats,
        "verification": check,
        "eager": es,
        "graphed": gs,
        "speedup": round(speedup, 4),
        "paired": paired,
    }
    slug = info["device_name"].lower().replace(" ", "-")
    out = Path(args.out or f"results/graphs_{args.model.split('/')[-1].lower()}_{slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
