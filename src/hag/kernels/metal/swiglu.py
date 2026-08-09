"""Fused SwiGLU as a custom Metal kernel, via MLX.

The Apple-silicon counterpart to `hag.kernels.triton.swiglu`. Same arithmetic,
same fusion, a different memory system underneath -- which is the entire point
of carrying both backends in one repo.

Note the asymmetry the two platforms create. On a discrete GPU the win comes
from not round-tripping a temporary through HBM. On unified memory there is no
separate device pool to round-trip through, so the win is narrower and comes
mostly from dispatch count and cache pressure. Measuring that difference is
more interesting than either number alone.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = """
    uint elem = thread_position_in_grid.x;
    // The dispatch is sized to exactly the element count, and Apple GPUs
    // support non-uniform threadgroups, so no bounds check is needed.
    float g = static_cast<float>(gate[elem]);
    float u = static_cast<float>(up[elem]);
    // sigmoid in fp32: silu saturates in fp16 well inside the range that real
    // gate activations occupy.
    float s = 1.0f / (1.0f + metal::exp(-g));
    out[elem] = static_cast<T>(g * s * u);
"""

_KERNEL = mx.fast.metal_kernel(
    name="hag_swiglu",
    input_names=["gate", "up"],
    output_names=["out"],
    source=_SOURCE,
)


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Fused `silu(gate) * up`. Mirrors `hag.reference.swiglu`."""
    if gate.shape != up.shape:
        raise ValueError(f"shape mismatch: {gate.shape} vs {up.shape}")
    if gate.dtype != up.dtype:
        raise ValueError(f"dtype mismatch: {gate.dtype} vs {up.dtype}")

    (out,) = _KERNEL(
        inputs=[gate, up],
        template=[("T", gate.dtype)],
        grid=(gate.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )
    return out


def swiglu_reference(gate: mx.array, up: mx.array) -> mx.array:
    """Unfused MLX baseline: what the framework does without a custom kernel."""
    return (gate * mx.sigmoid(gate)) * up
