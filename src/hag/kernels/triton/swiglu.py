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


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused `silu(gate) * up`. Matches `hag.reference.swiglu`."""
    assert gate.shape == up.shape, "gate and up must be the same shape"
    assert gate.is_cuda, "Triton kernels require a CUDA device"

    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    n = out.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_fwd[grid](gate, up, out, n)
    return out
