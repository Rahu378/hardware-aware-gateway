"""Drop the fused kernels into a HuggingFace model.

    import hag
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").cuda()
    hag.patch(model)

That is the whole integration. The kernels were always usable directly, but
they lived inside the benchmark, which meant anyone wanting to try them had to
read `bench_e2e` and copy the patching loop out of it.

`patch` is reversible. `unpatch(model)` restores the original bound methods,
which matters more than it sounds: the benchmark alternates between patched and
unpatched configurations inside a single process to cancel machine drift, and
that only works if the swap goes both ways exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Attribute on a patched module holding its original forward.
_ORIGINAL = "_hag_original_forward"


@dataclass
class PatchReport:
    """What was replaced, and what was left alone."""

    replaced: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.replaced)

    def __str__(self) -> str:
        if not self.replaced:
            return "hag: nothing patched"
        lines = [f"hag: patched {len(self.replaced)} modules"]
        lines += [f"  {name}" for name in sorted(set(self.replaced))]
        if self.skipped:
            lines.append(f"  left alone: {', '.join(sorted(set(self.skipped)))}")
        return "\n".join(lines)


def patch(model, verbose: bool = False) -> PatchReport:
    """Replace RMSNorm and the MLP activation with the fused Triton kernels.

    Attention is deliberately untouched. The profile that drove this work put
    matmul at 76% of decode GPU time and attention at roughly 1% for the
    sequence lengths measured, so there was nothing there worth taking.

    Returns a report rather than mutating silently, because the honest answer to
    "did this speed up my model" depends on which modules were actually hit, and
    a model whose class names do not match returns an empty report rather than
    an error.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The fused kernels are Triton, so they need a CUDA device. On Apple "
            "silicon use hag.kernels.metal directly; there is no patcher for it "
            "because MLX models are not nn.Module trees."
        )

    from .kernels.triton import rmsnorm as tri_rmsnorm
    from .kernels.triton import swiglu as tri_swiglu

    report = PatchReport()

    for module in model.modules():
        cls = type(module).__name__
        if hasattr(module, _ORIGINAL):
            continue

        if cls.endswith("RMSNorm") and hasattr(module, "weight"):
            eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))

            def rms_forward(self, hidden_states, _eps=eps):
                return tri_rmsnorm.rmsnorm(hidden_states, self.weight, _eps)

            setattr(module, _ORIGINAL, module.forward)
            module.forward = rms_forward.__get__(module, type(module))
            report.replaced.append(f"{cls}.forward -> triton.rmsnorm")

        elif cls.endswith("MLP") and all(
            hasattr(module, p) for p in ("gate_proj", "up_proj", "down_proj")
        ):

            def mlp_forward(self, x):
                # The dispatching entry point, not the raw kernel. Below
                # SWIGLU_MIN_ROWS it falls back to eager, which is what keeps
                # the prefill win without paying at decode.
                return self.down_proj(tri_swiglu.swiglu(self.gate_proj(x), self.up_proj(x)))

            setattr(module, _ORIGINAL, module.forward)
            module.forward = mlp_forward.__get__(module, type(module))
            report.replaced.append(
                f"{cls}.forward -> triton.swiglu (fused at >= "
                f"{tri_swiglu.SWIGLU_MIN_ROWS} rows)"
            )

        elif "Attention" in cls:
            report.skipped.append(cls)

    if verbose:
        print(report)
    return report


def unpatch(model) -> int:
    """Restore original forwards. Returns how many modules were restored."""
    count = 0
    for module in model.modules():
        original = getattr(module, _ORIGINAL, None)
        if original is not None:
            module.forward = original
            delattr(module, _ORIGINAL)
            count += 1
    return count


def is_patched(model) -> bool:
    return any(hasattr(m, _ORIGINAL) for m in model.modules())
