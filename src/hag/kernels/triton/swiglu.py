"""Fused SwiGLU activation in Triton.

`silu(gate) * up` is three PyTorch kernels and one full-size temporary:

    t = sigmoid(gate)     read gate,  write t
    t = gate * t          read gate, read t, write t
    out = t * up          read t, read up, write out

Seven array-sized transfers for an operation whose information content needs
three: read gate, read up, write out. On a 1.5B model the intermediate width is
8960, so at a 2048-token prefill that temporary is not small.

This kernel is where the memory-bound story is easiest to see, which is why it
is the one the report leads with.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=16),
    ],
    key=["n_elements"],
)
@triton.jit
def _swiglu_fwd(G, U, O, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # The sigmoid runs in fp32 regardless of storage dtype: silu saturates
    # badly in fp16 for |gate| beyond ~11, and gate activations do get there.
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    out = (g * tl.sigmoid(g)) * u
    tl.store(O + offs, out.to(O.dtype.element_ty), mask=mask)


#: Below this many rows, the eager path wins and this kernel dispatches to it.
#:
#: Fusing is a bandwidth optimisation, and bandwidth is not the constraint at a
#: single row. Measured on a T4: at 1 x 14336 the fused kernel takes 54 us
#: against eager's 16 us, because Triton's launch path costs more than the three
#: ATen launches it replaces while the traffic saved is a few kilobytes. At
#: 512 x 14336 the same kernel is 1.74x faster. The crossover sits between those
#: two, and `hag.bench_ops` sweeps 1/8/32/64/128/512/2048 rows and reports the
#: crossover directly, so this constant is set from a measurement rather than
#: taste. The current value is the midpoint of the bracket the first T4 sweep
#: established; rerun the sweep on new hardware and read `crossover_rows` from
#: the JSON before trusting it there.
#:
#: Note this threshold is specific to `swiglu`. `rmsnorm_residual` wins at every
#: row count measured, including 1, because it removes a whole round trip of the
#: hidden state rather than one intermediate, so it is not gated.
SWIGLU_MIN_ROWS = 64


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The fused kernel, unconditionally. Matches `hag.reference.swiglu`.

    Benchmarks call this directly so the sweep keeps reporting the kernel's real
    cost at every shape, including the shapes where it loses. Routing the
    benchmark through the dispatcher below would quietly replace those rows with
    measurements of eager against itself.
    """
    assert gate.shape == up.shape, "gate and up must be the same shape"
    assert gate.is_cuda, "Triton kernels require a CUDA device"

    if gate.dtype != up.dtype:
        raise ValueError(f"dtype mismatch: {gate.dtype} vs {up.dtype}")

    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    n = out.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_fwd[grid](gate, up, out, n)
    return out


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`silu(gate) * up`, fused when fusing is worth it.

    This is the entry point the model patcher uses. Both paths compute the same
    function to within a rounding of the final store; the choice is purely about
    which is faster at this shape.
    """
    rows = gate.numel() // gate.shape[-1] if gate.ndim else 0
    if rows < SWIGLU_MIN_ROWS:
        from ... import reference

        return reference.swiglu(gate, up)
    return swiglu_triton(gate, up)
