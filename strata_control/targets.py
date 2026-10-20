"""Which OS and backend combinations the project commits to, and what this machine is.

Pure and tiny on purpose: the engine resolver on the start path (engine_component.py) needs these answers, and the
start path must not import the component fetcher (components.py) to get them.
"""

from __future__ import annotations

import platform


class ComponentError(RuntimeError):
    """A component could not be used. The message names the reason and nothing was changed."""

    def __init__(self, message: str, code: int = 6) -> None:
        super().__init__(message)
        self.code = code


#: The OS × backend families the handoff commits to. A component may only be activated where its
#: family is listed here, so an unqualified combination is refused rather than attempted: the packet
#: includes this fork's native Windows SYCL engine; Intel Macs remain unsupported.
BACKEND_MATRIX: dict[str, tuple[str, ...]] = {
    "metal": ("macos-arm64",),
    "cuda": ("windows-x86_64", "linux-x86_64"),
    "hip": ("windows-x86_64", "linux-x86_64"),
    "sycl": ("windows-x86_64", "linux-x86_64"),
    "cpu": ("macos-arm64", "macos-x86_64", "windows-x86_64", "linux-x86_64"),
}

# Discrete Intel adapters supported by the native Windows installer (sycl/setup_intel.py).
WIN_INTEL_ARC_IDS = frozenset({0xe223, 0xe221, 0xe211, 0xe20b, 0xe20c, 0xe212,
                               0x56a0, 0x56a1, 0x56a2, 0x56a5, 0x56a6, 0x5690})


def host_platform() -> str:
    """This machine as an `os-arch` key, matching `BACKEND_MATRIX`."""
    system = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    if system is None:
        raise ComponentError(f"unsupported operating system: {platform.system()}")
    machine = platform.machine().lower()
    arch = {"arm64": "arm64", "aarch64": "arm64",
            "x86_64": "x86_64", "amd64": "x86_64"}.get(machine, machine)
    return f"{system}-{arch}"


def backend_supported(backend: str, target: str | None = None) -> tuple[bool, str]:
    """Whether a backend family is committed to on a platform, with the reason if not."""
    target = target or host_platform()
    if backend not in BACKEND_MATRIX:
        return False, f"unknown backend family {backend!r}"
    platforms = BACKEND_MATRIX[backend]
    if target in platforms:
        return True, ""
    return False, (f"{backend} is not a supported combination on {target}; it is supported on "
                   f"{', '.join(platforms)}. Unsupported combinations are not attempted.")
