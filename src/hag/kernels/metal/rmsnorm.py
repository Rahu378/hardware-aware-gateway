"""Fused residual-add + RMSNorm as a custom Metal kernel, via MLX.

One threadgroup per row. Each thread strides across the row accumulating a
partial sum of squares, the partials are reduced first within each SIMD group
(a register shuffle, no memory traffic) and then across SIMD groups through a
small threadgroup buffer.

The two-level reduction is the part worth reading. A naive implementation
reduces through threadgroup memory alone and pays a barrier per step; folding
the first 32 lanes into `simd_sum` removes five of those steps and leaves at
most `threads/32` partials for the slow path.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = """
#include <metal_simdgroup>
"""

_SOURCE = """
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const uint nthreads = threads_per_threadgroup.x;
    const uint base = row * N;

    // Pass 1: h = x + residual, written out, while accumulating sum of squares.
    float acc = 0.0f;
    for (uint i = lane; i < N; i += nthreads) {
        float v = static_cast<float>(x[base + i]) + static_cast<float>(res[base + i]);
        h[base + i] = static_cast<T>(v);
        acc += v * v;
    }

    // Two-level reduction: shuffle within the SIMD group, then across groups.
    // MLX injects Metal's built-in attributes by their canonical spelling, so
    // these are `thread_index_in_simdgroup` rather than the CUDA-flavoured
    // `simd_lane_id` an NVIDIA habit reaches for first.
    threadgroup float partials[32];
    acc = metal::simd_sum(acc);
    if (thread_index_in_simdgroup == 0) {
        partials[simdgroup_index_in_threadgroup] = acc;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    const uint n_simd_groups = (nthreads + 31) / 32;
    float total = 0.0f;
    for (uint i = 0; i < n_simd_groups; ++i) {
        total += partials[i];
    }
    const float rstd = metal::rsqrt(total / static_cast<float>(N) + eps);

    // Pass 2: re-read h. It was just written by this same threadgroup, so it is
    // still resident in cache; this costs far less than the loop shape implies.
    for (uint i = lane; i < N; i += nthreads) {
        float v = static_cast<float>(h[base + i]);
        y[base + i] = static_cast<T>(v * rstd * static_cast<float>(w[i]));
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="hag_rmsnorm_residual",
    input_names=["x", "res", "w", "eps"],
    output_names=["y", "h"],
    source=_SOURCE,
    header=_HEADER,
)

#: 256 threads keeps 8 SIMD groups busy without over-subscribing the reduction.
THREADS_PER_ROW = 256


def rmsnorm_residual(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """Fused (residual add -> RMSNorm). Returns `(normed, new_residual)`."""
    if x.shape != residual.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {residual.shape}")

    n = x.shape[-1]
    m = x.size // n
    if weight.shape != (n,):
        raise ValueError(f"weight must be ({n},), got {weight.shape}")

    y, h = _KERNEL(
        inputs=[x, residual, weight, mx.array(eps, dtype=mx.float32)],
        template=[("T", x.dtype), ("N", n)],
        grid=(m * THREADS_PER_ROW, 1, 1),
        threadgroup=(THREADS_PER_ROW, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return y, h


def rmsnorm_residual_reference(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """Unfused MLX baseline."""
    h = x + residual
    h32 = h.astype(mx.float32)
    var = mx.mean(h32 * h32, axis=-1, keepdims=True)
    y = h32 * mx.rsqrt(var + eps) * weight.astype(mx.float32)
    return y.astype(x.dtype), h
