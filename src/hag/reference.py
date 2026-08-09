"""PyTorch reference implementations.

These are the baselines every custom kernel is checked against, both for
numerical agreement and for speed. They are written the way the operation
appears in a stock Llama-family forward pass -- eager, unfused, one torch op at
a time -- because that is what the custom kernel is actually replacing.
"""

from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm as implemented in Llama / Qwen.

    The upcast to fp32 for the reduction is not optional: in fp16 the sum of
    squares over a 4096-wide hidden state overflows for perfectly ordinary
    activation magnitudes.
    """
    dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    y = x32 * torch.rsqrt(var + eps)
    return (y * weight.float()).to(dtype)


def rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The real decoder-block pattern: add the residual, then normalise.

    Returns `(normed, new_residual)`. The new residual is the *pre-norm* sum,
    which the block needs again after the attention/MLP sublayer -- which is
    why fusing these two ops saves a full round trip of the hidden state rather
    than just a kernel launch.
    """
    h = x + residual
    return rmsnorm(h, weight, eps), h


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: silu(gate) * up.

    In eager PyTorch this is two kernels and one full-size temporary.
    """
    return torch.nn.functional.silu(gate) * up


def ideal_bytes_rmsnorm(m: int, n: int, itemsize: int) -> int:
    """Minimum traffic: read x, write y."""
    return 2 * m * n * itemsize


def ideal_bytes_rmsnorm_residual(m: int, n: int, itemsize: int) -> int:
    """Minimum traffic: read x, read residual, write h, write y."""
    return 4 * m * n * itemsize


def ideal_bytes_swiglu(m: int, n: int, itemsize: int) -> int:
    """Minimum traffic: read gate, read up, write out."""
    return 3 * m * n * itemsize
