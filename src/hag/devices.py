"""Backend detection and vendor-published hardware limits.

The numbers in SPECS are *datasheet* figures, not measurements. They are only
used as a denominator when reporting "percent of peak". Every benchmark in this
repo also measures an achievable copy bandwidth on the live device
(`hag.microbench`), because the gap between datasheet and achievable is itself
part of the story.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    #: Datasheet peak DRAM / unified-memory bandwidth, GB/s (1 GB = 1e9 B).
    peak_bandwidth_gbs: float
    #: Datasheet dense half-precision throughput, TFLOP/s.
    peak_tflops_fp16: float
    #: On-package memory, GB.
    memory_gb: float
    #: Where the datasheet figure came from, so it can be re-checked.
    source: str


# Keyed by a substring match against the runtime device name.
SPECS: dict[str, DeviceSpec] = {
    "T4": DeviceSpec("NVIDIA T4", 320.0, 65.0, 16.0, "NVIDIA T4 datasheet"),
    "L4": DeviceSpec("NVIDIA L4", 300.0, 121.0, 24.0, "NVIDIA L4 datasheet"),
    "A10G": DeviceSpec("NVIDIA A10G", 600.0, 125.0, 24.0, "NVIDIA A10 datasheet"),
    "A100": DeviceSpec("NVIDIA A100 40GB", 1555.0, 312.0, 40.0, "NVIDIA A100 datasheet"),
    "V100": DeviceSpec("NVIDIA V100", 900.0, 125.0, 16.0, "NVIDIA V100 datasheet"),
    "M3": DeviceSpec("Apple M3", 100.0, 4.1, 8.0, "Apple M3 press specs (LPDDR5-6400, 128-bit)"),
    "M3 Pro": DeviceSpec("Apple M3 Pro", 150.0, 7.4, 18.0, "Apple M3 Pro press specs"),
    "M3 Max": DeviceSpec("Apple M3 Max", 400.0, 14.2, 36.0, "Apple M3 Max press specs"),
}


def available_backends() -> list[str]:
    """Backends usable in this process, most capable first."""
    found: list[str] = []
    try:
        import torch

        if torch.cuda.is_available():
            found.append("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            found.append("mps")
    except ImportError:
        pass
    try:
        import mlx.core  # noqa: F401

        found.append("mlx")
    except ImportError:
        pass
    found.append("cpu")
    return found


def default_backend() -> str:
    return available_backends()[0]


def device_name(backend: str) -> str:
    if backend == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    if backend in ("mps", "mlx"):
        return _apple_chip_name()
    return platform.processor() or platform.machine()


def _apple_chip_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.machine()


def lookup_spec(name: str) -> DeviceSpec | None:
    """Best-effort match of a runtime device name against SPECS.

    Longest key first so "M3 Max" wins over "M3".
    """
    for key in sorted(SPECS, key=len, reverse=True):
        if key.lower() in name.lower():
            return SPECS[key]
    return None


def describe(backend: str | None = None) -> dict:
    backend = backend or default_backend()
    name = device_name(backend)
    spec = lookup_spec(name)
    return {
        "backend": backend,
        "device_name": name,
        "spec": None if spec is None else spec.__dict__,
        "platform": platform.platform(),
    }
