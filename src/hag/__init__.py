"""Hardware-Aware Gateway: profile-driven kernel work on LLM inference.

The package is deliberately split so that nothing imports a backend it does not
need. Importing `hag` on a machine with no CUDA and no Metal is expected to work
and is covered by the test suite.
"""

__version__ = "0.1.0"

# The implementations live in `hag.integrate` and `hag.autotune` rather than
# modules named after these functions. A module and a function of the same name
# on the same package collide: `import hag.patch` rebinds `hag.patch` from the
# function to the module, so which one you get depends on what else was
# imported first.


def patch(model, verbose: bool = False):
    """Swap the fused kernels into a HuggingFace model. See `hag.integrate`."""
    from .integrate import patch as _patch

    return _patch(model, verbose=verbose)


def unpatch(model) -> int:
    from .integrate import unpatch as _unpatch

    return _unpatch(model)


def calibrate(force: bool = False):
    """Measure this device's fusion crossover. See `hag.autotune`."""
    from .autotune import calibrate as _calibrate

    return _calibrate(force=force)


__all__ = ["__version__", "calibrate", "patch", "unpatch"]
