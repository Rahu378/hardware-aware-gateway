"""Fused residual-add + RMSNorm in Triton.

The op being replaced is the top of every Llama-family decoder block:

    h = x + residual          # one full read-read-write of the hidden state
    y = rmsnorm(h) * weight   # another full read, plus a write

Eager PyTorch runs that as four separate kernels and materialises `h` to DRAM
before reading it straight back. Fusing collapses it to a single pass: read `x`,
read `residual`, write `h`, write `y`. That is 4 arrays of traffic against the
7-ish the unfused chain moves, and at decode time -- when the GPU is idle
waiting on memory, not on maths -- traffic is the only thing that matters.

Forward only. This is an inference project; there is no backward pass to write.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_residual_1pass(
    X, RES, W, Y, H,
    stride,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Specialisation for rows that fit in one block.

    `h` never leaves registers between the reduction and the scaling, so the
    hidden state is read from DRAM exactly once. Every hidden size we care
    about (2048, 3072, 4096, 8192) fits here.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * stride + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + off, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    tl.store(H + off, h.to(H.dtype.element_ty), mask=mask)

    # Written as 1/sqrt rather than tl.rsqrt: the spelling of the reciprocal
    # square root moved between Triton 2.x and 3.x, and this form compiles to
    # the same instruction on both.
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + off, (h * rstd * w).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_residual_looped(
    X, RES, W, Y, H,
    stride,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    """General fallback for hidden sizes too wide to hold in registers.

    Costs a second read of `h`, but that read hits L2 rather than DRAM, so the
    penalty is far smaller than the shape of the code suggests.
    """
    row = tl.program_id(0)
    base = row * stride

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RES + base + cols, mask=mask, other=0.0).to(tl.float32)
        h = x + r
        tl.store(H + base + cols, h.to(H.dtype.element_ty), mask=mask)
        acc += h * h
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)

    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        h = tl.load(H + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + base + cols, (h * rstd * w).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_plain(
    X, W, Y,
    stride,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """RMSNorm without the residual add.

    Exists as a separate kernel rather than the fused one with a zeroed
    residual, because that trick would move an extra array of zeros through
    DRAM to save a hundred lines of source -- the wrong trade in a kernel whose
    entire purpose is traffic reduction. This is the drop-in replacement for
    `LlamaRMSNorm.forward` used by the end-to-end benchmark.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * stride + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + off, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm. Matches `hag.reference.rmsnorm`."""
    assert x.is_cuda, "Triton kernels require a CUDA device"
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, n = x2d.shape
    if n > ONE_PASS_LIMIT:
        raise ValueError(
            f"hidden size {n} exceeds the single-block path; use rmsnorm_residual"
        )
    y = torch.empty_like(x2d)
    block_n = triton.next_power_of_2(n)
    _rmsnorm_plain[(m,)](
        x2d, weight, y, x2d.stride(0), n, eps,
        BLOCK_N=block_n, num_warps=_num_warps_for(block_n),
    )
    return y.reshape(x.shape)


def _num_warps_for(block_n: int) -> int:
    # More warps only help once there is enough work per row to hide the
    # reduction; past 8 the tree-reduction latency starts to dominate.
    if block_n >= 8192:
        return 16
    if block_n >= 4096:
        return 8
    if block_n >= 1024:
        return 4
    return 2


#: Above this width a row no longer fits comfortably in registers on an SM.
ONE_PASS_LIMIT = 16384


def rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused (residual add -> RMSNorm). Returns `(normed, new_residual)`.

    Matches `hag.reference.rmsnorm_residual` including its fp32 reduction.
    """
    assert x.shape == residual.shape, "residual must match the hidden state"
    assert x.is_cuda, "Triton kernels require a CUDA device"

    x2d = x.reshape(-1, x.shape[-1])
    res2d = residual.reshape(-1, residual.shape[-1])
    # A non-contiguous row stride would silently corrupt the flat indexing above.
    x2d = x2d.contiguous()
    res2d = res2d.contiguous()

    m, n = x2d.shape
    y = torch.empty_like(x2d)
    h = torch.empty_like(x2d)

    if n <= ONE_PASS_LIMIT:
        block_n = triton.next_power_of_2(n)
        _rmsnorm_residual_1pass[(m,)](
            x2d, res2d, weight, y, h,
            x2d.stride(0), n, eps,
            BLOCK_N=block_n,
            num_warps=_num_warps_for(block_n),
        )
    else:
        block_n = 4096
        _rmsnorm_residual_looped[(m,)](
            x2d, res2d, weight, y, h,
            x2d.stride(0), n, eps,
            BLOCK_N=block_n,
            num_warps=_num_warps_for(block_n),
        )

    return y.reshape(x.shape), h.reshape(x.shape)
