"""Numerical agreement between every custom kernel and its reference.

A kernel that is fast and wrong is worth nothing, so this suite runs before any
benchmark in the Makefile. Each backend is skipped rather than failed when the
hardware is absent, so the same file runs on a Mac, on a CUDA box, and in CI
with no GPU at all.

Tolerances are set per dtype against the *reference*, not against an exact
answer. Both implementations reduce in fp32 and store in the working dtype, so
the residual disagreement should be a single rounding of the final store.
"""

from __future__ import annotations

import pytest

from hag import devices

BACKENDS = devices.available_backends()
HAS_CUDA = "cuda" in BACKENDS
HAS_MLX = "mlx" in BACKENDS

SHAPES = [(1, 2048), (8, 1536), (7, 3072), (512, 4096), (2048, 2048)]

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="no Apple Metal device")


def _tol(dtype_name: str) -> float:
    # fp16 has ~3 decimal digits; one rounding of the final store is ~1e-2 for
    # activations of order 1. fp32 should agree to within a few ulps.
    return {"fp16": 2e-2, "bf16": 1e-1, "fp32": 1e-5}[dtype_name]


# --------------------------------------------------------------------------
# Triton / CUDA
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype_name", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_rmsnorm_residual(dtype_name, shape):
    import torch

    from hag import reference
    from hag.kernels.triton import rmsnorm as tri

    dt = {"fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    m, n = shape
    x = torch.randn(m, n, dtype=dt, device="cuda")
    res = torch.randn(m, n, dtype=dt, device="cuda")
    w = torch.randn(n, dtype=dt, device="cuda")

    y, h = tri.rmsnorm_residual(x, res, w)
    y_ref, h_ref = reference.rmsnorm_residual(x, res, w)

    torch.testing.assert_close(h, h_ref, atol=_tol(dtype_name), rtol=_tol(dtype_name))
    torch.testing.assert_close(y, y_ref, atol=_tol(dtype_name), rtol=_tol(dtype_name))


@requires_cuda
@pytest.mark.parametrize("dtype_name", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_swiglu(dtype_name, shape):
    import torch

    from hag import reference
    from hag.kernels.triton import swiglu as tri

    dt = {"fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    m, n = shape
    g = torch.randn(m, n, dtype=dt, device="cuda")
    u = torch.randn(m, n, dtype=dt, device="cuda")

    torch.testing.assert_close(
        tri.swiglu_triton(g, u), reference.swiglu(g, u),
        atol=_tol(dtype_name), rtol=_tol(dtype_name),
    )


@requires_cuda
@pytest.mark.parametrize("rows", [1, 8, 63, 64, 512])
def test_triton_swiglu_dispatcher_matches_reference(rows):
    """The dispatcher must be transparent: same answer either side of the threshold."""
    import torch

    from hag import reference
    from hag.kernels.triton import swiglu as tri

    g = torch.randn(rows, 8960, dtype=torch.float16, device="cuda")
    u = torch.randn(rows, 8960, dtype=torch.float16, device="cuda")
    torch.testing.assert_close(tri.swiglu(g, u), reference.swiglu(g, u), atol=2e-2, rtol=2e-2)


@requires_cuda
def test_triton_swiglu_dispatcher_routes_on_row_count():
    """Below the threshold it must actually take the eager path, not just agree.

    Asserting only on numerical agreement would pass even if the threshold were
    ignored, since both paths compute the same function.
    """
    import torch

    from hag.kernels.triton import swiglu as tri

    calls = []
    original = tri.swiglu_triton
    tri.swiglu_triton = lambda g, u: calls.append(1) or original(g, u)
    try:
        below = torch.randn(tri.SWIGLU_MIN_ROWS - 1, 512, dtype=torch.float16, device="cuda")
        tri.swiglu(below, below)
        assert not calls, "should have used the eager path below the threshold"

        at = torch.randn(tri.SWIGLU_MIN_ROWS, 512, dtype=torch.float16, device="cuda")
        tri.swiglu(at, at)
        assert calls, "should have used the fused kernel at the threshold"
    finally:
        tri.swiglu_triton = original


@requires_cuda
def test_triton_swiglu_saturating_inputs():
    """silu must not blow up where fp16 would if the sigmoid ran in half."""
    import torch

    from hag import reference
    from hag.kernels.triton import swiglu as tri

    g = torch.tensor([[-30.0, -12.0, 0.0, 12.0, 30.0]], dtype=torch.float16, device="cuda")
    u = torch.ones_like(g)
    out = tri.swiglu_triton(g, u)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, reference.swiglu(g, u), atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------
# Metal / MLX
# --------------------------------------------------------------------------


@requires_mlx
@pytest.mark.parametrize("dtype_name", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_metal_rmsnorm_residual(dtype_name, shape):
    import mlx.core as mx

    from hag.kernels.metal import rmsnorm as mtl

    dt = {"fp16": mx.float16, "fp32": mx.float32}[dtype_name]
    m, n = shape
    x = mx.random.normal((m, n)).astype(dt)
    res = mx.random.normal((m, n)).astype(dt)
    w = mx.random.normal((n,)).astype(dt)

    y, h = mtl.rmsnorm_residual(x, res, w)
    y_ref, h_ref = mtl.rmsnorm_residual_reference(x, res, w)
    mx.eval(y, h, y_ref, h_ref)

    tol = _tol(dtype_name)
    assert mx.allclose(h.astype(mx.float32), h_ref.astype(mx.float32), atol=tol, rtol=tol)
    assert mx.allclose(y.astype(mx.float32), y_ref.astype(mx.float32), atol=tol, rtol=tol)


@requires_mlx
@pytest.mark.parametrize("dtype_name", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_metal_swiglu(dtype_name, shape):
    import mlx.core as mx

    from hag.kernels.metal import swiglu as mtl

    dt = {"fp16": mx.float16, "fp32": mx.float32}[dtype_name]
    m, n = shape
    g = mx.random.normal((m, n)).astype(dt)
    u = mx.random.normal((m, n)).astype(dt)

    out = mtl.swiglu(g, u)
    ref = mtl.swiglu_reference(g, u)
    mx.eval(out, ref)

    tol = _tol(dtype_name)
    assert mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=tol, rtol=tol)


@requires_mlx
def test_metal_rmsnorm_rejects_bad_weight():
    import mlx.core as mx

    from hag.kernels.metal import rmsnorm as mtl

    x = mx.random.normal((4, 128))
    with pytest.raises(ValueError):
        mtl.rmsnorm_residual(x, x, mx.ones((64,)))


# --------------------------------------------------------------------------
# Backend-independent
# --------------------------------------------------------------------------


def test_package_imports_without_any_gpu():
    """`import hag` must not require a backend. CI has no GPU."""
    import hag
    from hag import devices, reference, timing  # noqa: F401

    assert hag.__version__
    assert "cpu" in devices.available_backends()


def test_ideal_byte_counts():
    from hag import reference

    # fp16, 2048 rows of 4096: read x, read residual, write h, write y.
    assert reference.ideal_bytes_rmsnorm_residual(2048, 4096, 2) == 4 * 2048 * 4096 * 2
    # read gate, read up, write out.
    assert reference.ideal_bytes_swiglu(2048, 4096, 2) == 3 * 2048 * 4096 * 2
